#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from fractions import Fraction
from pathlib import Path

import av
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC_DIRS = (
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
)
for src_dir in PACKAGE_SRC_DIRS:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines import ICLoraPipeline
from ltx_pipelines.stage2_routing import load_stage2_input_cache
from ltx_pipelines.correspondence_mask import (
    CorrespondenceMaskBox,
    build_box_mask,
    build_later_frames_mask,
    load_file_mask,
)
from ltx_pipelines.utils.args import (
    QUANTIZATION_POLICIES,
    ImageConditioningInput,
    QuantizationAction,
    _PipelineArgumentParser,
)
from ltx_pipelines.utils.attention_probe import AttentionProbe, AttentionProbeConfig, parse_index_set
from ltx_pipelines.utils.conditioning_schedules import (
    ConstantSourceStrengthSchedule,
    SourceStrengthRouting,
    build_first_frame_attention_schedule,
    build_source_strength_schedule,
    parse_first_frame_attention_values,
    parse_source_strength_values,
)
from ltx_pipelines.utils.media_io import encode_video


class ImageConditioningAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (3, 4):
            raise argparse.ArgumentError(
                self,
                f"{option_string} requires PATH FRAME_IDX STRENGTH [CRF], got {len(values)} values",
            )
        conditioning = ImageConditioningInput(
            path=str(Path(values[0]).expanduser().resolve()),
            frame_idx=int(values[1]),
            strength=float(values[2]),
            crf=int(values[3]) if len(values) == 4 else 33,
        )
        current = getattr(namespace, self.dest) or []
        current.append(conditioning)
        setattr(namespace, self.dest, current)


def parse_args() -> argparse.Namespace:
    parser = _PipelineArgumentParser(
        description=(
            "Run LTX-2 IC-LoRA style transfer: preserve the motion/camera/structure of a "
            "reference video while steering appearance toward a reference image."
        )
    )
    parser.add_argument("--distilled-checkpoint-path", required=True, help="Path to the distilled LTX-2 checkpoint.")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Path to the spatial upsampler checkpoint.")
    parser.add_argument("--gemma-root", required=True, help="Path to the Gemma text-encoder directory.")
    parser.add_argument("--ic-lora-path", required=True, help="Path to the IC-LoRA weights.")
    parser.add_argument(
        "--reference-video", required=True, help="Reference video whose motion and camera should be followed."
    )
    parser.add_argument(
        "--conditioning-video",
        required=True,
        help=(
            "Precomputed conditioning/control video. Prepare this with "
            "style_transfer_preprocess.py or translate_single_sequence.py."
        ),
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference image whose style/appearance should influence the output.",
    )
    parser.add_argument(
        "--no-image-conditioning",
        action="store_true",
        help=(
            "Disable all reference-image conditioning. When enabled, no default image condition "
            "is created and Stage 2 runs without an image anchor."
        ),
    )
    parser.add_argument(
        "--image-condition",
        action=ImageConditioningAction,
        nargs="+",
        default=None,
        metavar="IMAGE_CONDITION",
        help=(
            "Additional/replacement image conditioning. Can be repeated. "
            "When omitted, --reference-image/--image-frame-idx/--image-strength/--image-crf are used."
        ),
    )
    parser.add_argument(
        "--reference-image-replace",
        type=int,
        nargs="*",
        default=None,
        metavar="FRAME_IDX",
        help=(
            "Pixel-frame indices whose image conditions should replace target latents in place. "
            "When omitted, frame 0 keeps the existing in-place replacement behavior. "
            "Pass the option without values to disable in-place image replacement."
        ),
    )
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--prompt",
        default=(
            "Follow the composition, timing, camera movement, and scene dynamics of the reference video, "
            "but render the result in the visual style, materials, colors, and identity implied by the reference image."
        ),
        help="Prompt to pair with the video and image conditioning.",
    )
    parser.add_argument(
        "--negative-prompt", default="", help="Optional negative prompt text appended to the main prompt."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. Defaults to the reference video height rounded down to a multiple of 64.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. Defaults to the reference video width rounded down to a multiple of 64.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to generate. Defaults to the full reference video length.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Output frame rate. Defaults to the reference video FPS.",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index where the reference image is injected as image conditioning.",
    )
    parser.add_argument(
        "--image-strength", type=float, default=1.0, help="Strength for the reference image conditioning."
    )
    parser.add_argument(
        "--image-crf", type=int, default=33, help="CRF-like preprocessing parameter for image conditioning."
    )
    parser.add_argument(
        "--video-strength", type=float, default=1.0, help="Strength for the reference video IC-LoRA conditioning."
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=("rgb", "depth", "edge", "rgb+depth", "edge+rgb", "edge+depth+rgb"),
        default="rgb",
        help=(
            "Interpretation of the conditioning video(s) expected by the IC-LoRA. "
            "Multi-signal modes: 'rgb+depth' (--conditioning-video + --depth-conditioning-video), "
            "'edge+rgb' (--conditioning-video + --edge-conditioning-video), "
            "'edge+depth+rgb' (all three)."
        ),
    )
    parser.add_argument(
        "--depth-conditioning-video",
        default=None,
        help="Depth conditioning video. Required for rgb+depth and edge+depth+rgb modes.",
    )
    parser.add_argument(
        "--edge-conditioning-video",
        default=None,
        help="Edge conditioning video. Required for edge+rgb and edge+depth+rgb modes.",
    )
    parser.add_argument(
        "--rgb-strength",
        type=float,
        default=None,
        help="Strength for the RGB condition in multi-signal modes. Defaults to --video-strength.",
    )
    parser.add_argument(
        "--depth-strength",
        type=float,
        default=None,
        help="Strength for the depth condition in multi-signal modes. Defaults to --video-strength.",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=None,
        help="Strength for the edge condition in multi-signal modes. Defaults to --video-strength.",
    )
    parser.add_argument(
        "--conditioning-attention-strength",
        type=float,
        default=1.0,
        help="How strongly the reference video conditioning should influence attention, in [0, 1].",
    )
    parser.add_argument(
        "--source-strength-schedule",
        choices=("constant", "values", "sigma-ramp"),
        default="constant",
        help="Per-transformer-evaluation schedule for IC-LoRA source-video attention.",
    )
    parser.add_argument(
        "--source-strength",
        type=float,
        default=1.0,
        help="Constant source attention strength for --source-strength-schedule constant.",
    )
    parser.add_argument(
        "--source-strength-values",
        default=None,
        help="Comma-separated per-evaluation source attention strengths for --source-strength-schedule values.",
    )
    parser.add_argument(
        "--source-strength-early",
        type=float,
        default=1.0,
        help="Early source attention strength for --source-strength-schedule sigma-ramp.",
    )
    parser.add_argument(
        "--source-strength-late",
        type=float,
        default=0.30,
        help="Late source attention strength for --source-strength-schedule sigma-ramp.",
    )
    parser.add_argument(
        "--source-strength-fade-start-sigma",
        type=float,
        default=0.725,
        help="Sigma where sigma-ramp starts fading source attention.",
    )
    parser.add_argument(
        "--source-strength-fade-end-sigma",
        type=float,
        default=0.421875,
        help="Sigma where sigma-ramp reaches late source attention.",
    )
    parser.add_argument(
        "--source-strength-routing",
        choices=("symmetric", "target-queries-only"),
        default="symmetric",
        help="Attention blocks affected by the source strength schedule.",
    )
    parser.add_argument(
        "--source-strength-log",
        default=None,
        help="Optional JSON path for resolved per-evaluation source strength rows.",
    )
    parser.add_argument(
        "--first-frame-attention-schedule",
        choices=("constant", "values", "sigma-ramp"),
        default="constant",
        help="Per-transformer-evaluation multiplier for later target tokens attending to frame-0 target tokens.",
    )
    parser.add_argument(
        "--first-frame-attention-multiplier",
        type=float,
        default=1.0,
        help="Constant first-frame attention multiplier for --first-frame-attention-schedule constant.",
    )
    parser.add_argument(
        "--first-frame-attention-values",
        default=None,
        help="Comma-separated per-evaluation first-frame attention multipliers.",
    )
    parser.add_argument(
        "--first-frame-attention-early",
        type=float,
        default=1.0,
        help="Early first-frame attention multiplier for --first-frame-attention-schedule sigma-ramp.",
    )
    parser.add_argument(
        "--first-frame-attention-late",
        type=float,
        default=4.0,
        help="Late first-frame attention multiplier for --first-frame-attention-schedule sigma-ramp.",
    )
    parser.add_argument(
        "--first-frame-attention-fade-start-sigma",
        type=float,
        default=0.975,
        help="Sigma where first-frame attention starts ramping up.",
    )
    parser.add_argument(
        "--first-frame-attention-fade-end-sigma",
        type=float,
        default=0.421875,
        help="Sigma where first-frame attention reaches its late multiplier.",
    )
    parser.add_argument(
        "--first-frame-attention-max",
        type=float,
        default=16.0,
        help="Validation cap for first-frame attention multipliers.",
    )
    parser.add_argument(
        "--first-frame-attention-log",
        default=None,
        help="Optional JSON path for resolved per-evaluation first-frame attention rows.",
    )
    parser.add_argument(
        "--stage-2-ic-lora-strength",
        type=float,
        default=0.0,
        help=(
            "IC-LoRA adapter strength in Stage 2. Zero preserves the previous image-only Stage 2; "
            "a positive value also appends the source-video reference tokens in Stage 2."
        ),
    )
    parser.add_argument(
        "--stage-2-conditioning-attention-strength",
        type=float,
        default=1.0,
        help="Stage-2 source-video conditioning attention strength, in [0, 1].",
    )
    parser.add_argument(
        "--stage-2-branch-mode",
        choices=("legacy", "image", "video", "global", "spatial"),
        default="legacy",
        help="Stage-2 branch execution mode. Legacy preserves the existing pre-fused path.",
    )
    parser.add_argument(
        "--stage-2-video-mix",
        type=float,
        default=0.5,
        help="Video-branch weight for global Stage-2 prediction routing.",
    )
    parser.add_argument(
        "--stage-2-routing-mask",
        default=None,
        help="Preprocessed dress mask used by spatial Stage-2 routing (M=1 in the dress region).",
    )
    parser.add_argument(
        "--stage-2-dress-video-contribution",
        type=float,
        default=None,
        help="Video-branch contribution inside the dress mask for spatial routing.",
    )
    parser.add_argument(
        "--stage-2-noise-seed",
        type=int,
        default=None,
        help="Independent Stage-2 noise seed. Defaults to --seed.",
    )
    parser.add_argument(
        "--stage-2-prediction-dir",
        default=None,
        help="Optional directory for per-step image/video/routed X0 safetensors and metrics.",
    )
    parser.add_argument(
        "--correspondence-mask-mode",
        choices=("later-frames", "boxes", "file"),
        default="later-frames",
        help=(
            "Mask source for target-to-source attention bias. The default biases all pixels after frame 0; "
            "'boxes' uses --correspondence-mask-box and 'file' uses --correspondence-mask-file."
        ),
    )
    parser.add_argument(
        "--correspondence-mask-file",
        default=None,
        help="Mask video, image, PNG directory, .npy, or .pt used with --correspondence-mask-mode file.",
    )
    parser.add_argument(
        "--correspondence-mask-box",
        type=float,
        nargs=6,
        action="append",
        default=None,
        metavar=("START", "END", "X0", "Y0", "X1", "Y1"),
        help=(
            "Frame-inclusive box with normalized coordinates, used with mask mode 'boxes'. "
            "May be repeated. START and END must be integer-valued."
        ),
    )
    parser.add_argument(
        "--correspondence-mask-feather",
        type=int,
        default=0,
        help="Box-mask feather radius in Stage-1 pixels. Default: 0.",
    )
    parser.add_argument(
        "--source-correspondence-bias",
        type=float,
        default=0.0,
        help="Additive pre-softmax target-to-source correspondence bias. Zero disables the feature.",
    )
    parser.add_argument(
        "--source-correspondence-radius",
        type=int,
        default=0,
        help="Source-latent neighborhood radius receiving the correspondence bias. Default: 0.",
    )
    parser.add_argument(
        "--stage-2-masked-denoise-strength",
        "--stage-2-masked-noise-strength",
        type=float,
        default=1.0,
        help=(
            "Per-token Stage-2 denoising strength inside the correspondence mask. "
            "One preserves current behavior; zero locks those tokens to the upsampled Stage-1 latent."
        ),
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Skip the upsampling/refinement stage for faster, lower-resolution output.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Let the text encoder enhance the prompt using the reference image as context.",
    )
    parser.add_argument(
        "--streaming-prefetch-count",
        type=int,
        default=None,
        help="Enable layer-streaming prefetching if supported by your setup.",
    )
    parser.add_argument(
        "--quiet-layer-streaming",
        action="store_true",
        help="Suppress warning-level layer-streaming memory logs.",
    )
    parser.add_argument(
        "--attention-probe-output",
        default=None,
        help="Optional JSONL path for aggregate video self-attention probing metrics.",
    )
    parser.add_argument(
        "--attention-probe-layers",
        default="0,8,16,24,32,40,47",
        help="Comma-separated layer indices or ranges to probe, or 'all'. Default: 0,8,16,24,32,40,47.",
    )
    parser.add_argument(
        "--attention-probe-steps",
        default="all",
        help="Comma-separated denoising step indices or ranges to probe, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--attention-probe-heads",
        choices=("mean", "all"),
        default="mean",
        help="Record mean metrics across heads or one JSONL record per head.",
    )
    parser.add_argument(
        "--attention-probe-query-chunk-size",
        type=int,
        default=128,
        help="Number of target query tokens per probing chunk. Lower values use less memory.",
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action=QuantizationAction,
        nargs="+",
        metavar=("POLICY", "AMAX_PATH"),
        default=None,
        help=(
            f"Quantization policy: {', '.join(QUANTIZATION_POLICIES)}. "
            "Example: --quantization fp8-cast or --quantization fp8-scaled-mm /path/to/amax.json"
        ),
    )
    parser.add_argument(
        "--save-stage-2-input",
        default=None,
        help="Save the upsampled Stage-1 video and audio latents for controlled Stage-2 reuse.",
    )
    parser.add_argument(
        "--load-stage-2-input",
        default=None,
        help="Load a validated Stage-2 input cache and skip Stage 1.",
    )
    parser.add_argument(
        "--initial-video-latent",
        default=None,
        help="Optional path to a saved Stage 1 video latent tensor (.pt) used instead of Gaussian noise.",
    )
    return parser.parse_args()


def read_video_metadata(video_path: str) -> tuple[int, int, int, float]:
    path = Path(video_path).expanduser().resolve()
    with av.open(str(path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {path}")

        width = int(video_stream.codec_context.width or video_stream.width)
        height = int(video_stream.codec_context.height or video_stream.height)

        fps_value = video_stream.average_rate or video_stream.base_rate or Fraction(25, 1)
        fps = float(fps_value)

        frames = video_stream.frames
        if not frames or frames <= 0:
            frames = 0
            for _packet in container.demux(video_stream):
                for _frame in _packet.decode():
                    frames += 1

    if frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")
    return width, height, frames, fps


def round_down_to_multiple(value: int, divisor: int) -> int:
    rounded = value - (value % divisor)
    if rounded <= 0:
        raise ValueError(f"Value {value} is too small to round down to a positive multiple of {divisor}")
    return rounded


def build_prompt(prompt: str, negative_prompt: str) -> str:
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    if not negative_prompt:
        return prompt
    return f"{prompt}\n\nNegative prompt: {negative_prompt}"


def is_readable_video_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False

    try:
        with av.open(str(path)) as container:
            return any(stream.type == "video" for stream in container.streams)
    except Exception:
        return False


def resolve_conditioning_videos(args: argparse.Namespace) -> list[tuple[str, float]]:
    def _validated(label: str, path_str: str | None, flag: str) -> Path:
        if not path_str:
            raise ValueError(f"{flag} is required when --conditioning-mode {args.conditioning_mode} is used.")
        p = Path(path_str).expanduser().resolve()
        if not is_readable_video_file(p):
            raise ValueError(
                f"{label} conditioning video does not exist or is unreadable: {p}. "
                "Prepare it with style_transfer_preprocess.py or translate_single_sequence.py."
            )
        return p

    def _s(attr: str) -> float:
        v = getattr(args, attr, None)
        return v if v is not None else args.video_strength

    mode = args.conditioning_mode

    if mode in ("rgb+depth", "edge+rgb", "edge+depth+rgb"):
        rgb_path = _validated("RGB", args.conditioning_video, "--conditioning-video")
        rgb_str = _s("rgb_strength")
        print(f"Using RGB conditioning video: {rgb_path} (strength={rgb_str})")
        result = [(str(rgb_path), rgb_str)]

        if mode in ("rgb+depth", "edge+depth+rgb"):
            depth_path = _validated("depth", args.depth_conditioning_video, "--depth-conditioning-video")
            depth_str = _s("depth_strength")
            print(f"Using depth conditioning video: {depth_path} (strength={depth_str})")
            result.append((str(depth_path), depth_str))

        if mode in ("edge+rgb", "edge+depth+rgb"):
            edge_path = _validated("edge", args.edge_conditioning_video, "--edge-conditioning-video")
            edge_str = _s("edge_strength")
            print(f"Using edge conditioning video: {edge_path} (strength={edge_str})")
            result.append((str(edge_path), edge_str))

        return result

    # Single-video modes
    conditioning_path = _validated("conditioning", args.conditioning_video, "--conditioning-video")
    print(f"Using conditioning video: {conditioning_path}")
    return [(str(conditioning_path), args.video_strength)]


def resolve_image_conditionings(args: argparse.Namespace, num_frames: int) -> list[ImageConditioningInput]:
    if args.no_image_conditioning:
        if args.image_condition:
            raise ValueError("--no-image-conditioning cannot be combined with --image-condition.")
        return []

    if not args.reference_image:
        raise ValueError("--reference-image is required unless --no-image-conditioning is used.")

    image_conditionings = args.image_condition or [
        ImageConditioningInput(
            path=str(Path(args.reference_image).expanduser().resolve()),
            frame_idx=args.image_frame_idx,
            strength=args.image_strength,
            crf=args.image_crf,
        )
    ]
    for conditioning in image_conditionings:
        if conditioning.frame_idx < 0 or conditioning.frame_idx >= num_frames:
            raise ValueError(f"image condition frame_idx must be in [0, {num_frames - 1}]: {conditioning}")
    return image_conditionings


def resolve_reference_image_replace(
    args: argparse.Namespace,
    image_conditionings: list[ImageConditioningInput],
    num_frames: int,
) -> set[int] | None:
    if args.reference_image_replace is None:
        return None

    replace_frame_indices = sorted(set(args.reference_image_replace))
    out_of_range = [idx for idx in replace_frame_indices if idx < 0 or idx >= num_frames]
    if out_of_range:
        raise ValueError(f"reference-image-replace frame index must be in [0, {num_frames - 1}]: {out_of_range}")

    image_frame_indices = {conditioning.frame_idx for conditioning in image_conditionings}
    missing = [idx for idx in replace_frame_indices if idx not in image_frame_indices]
    if missing:
        raise ValueError(
            "--reference-image-replace can only name frames that have image conditions. "
            f"Missing image conditions for frames: {missing}. "
            "Add matching --image-condition entries or include the frames in --reference-frame-list."
        )

    return set(replace_frame_indices)


def _asset_signature(path: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_stage_2_cache_metadata(
    args: argparse.Namespace,
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    image_conditionings: list[ImageConditioningInput],
    video_conditioning: list[tuple[str, float]],
) -> dict[str, object]:
    config = {
        "prompt": prompt,
        "seed": args.seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "checkpoint": _asset_signature(args.distilled_checkpoint_path),
        "ic_lora": _asset_signature(args.ic_lora_path),
        "reference_video": _asset_signature(args.reference_video),
        "video_conditioning": [
            {"asset": _asset_signature(path), "strength": strength} for path, strength in video_conditioning
        ],
        "image_conditioning": [
            {
                "asset": _asset_signature(item.path),
                "frame_idx": item.frame_idx,
                "strength": item.strength,
                "crf": item.crf,
            }
            for item in image_conditionings
        ],
        "conditioning_attention_strength": args.conditioning_attention_strength,
        "source_strength": (
            {"schedule": "constant", "constant": 1.0, "routing": args.source_strength_routing}
            if args.stage_2_branch_mode != "legacy"
            else {
                "schedule": args.source_strength_schedule,
                "constant": args.source_strength,
                "values": args.source_strength_values,
                "early": args.source_strength_early,
                "late": args.source_strength_late,
                "fade_start": args.source_strength_fade_start_sigma,
                "fade_end": args.source_strength_fade_end_sigma,
                "routing": args.source_strength_routing,
            }
        ),
        "first_frame_attention": {
            "schedule": args.first_frame_attention_schedule,
            "multiplier": args.first_frame_attention_multiplier,
            "values": args.first_frame_attention_values,
            "early": args.first_frame_attention_early,
            "late": args.first_frame_attention_late,
            "fade_start": args.first_frame_attention_fade_start_sigma,
            "fade_end": args.first_frame_attention_fade_end_sigma,
        },
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {"cache_version": 1, "stage_1_fingerprint": fingerprint, "stage_1_config": config}


def main() -> None:
    args = parse_args()

    if args.quiet_layer_streaming:
        logging.getLogger("ltx_core.layer_streaming").setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run LTX-2 IC-LoRA inference.")
    torch.cuda.reset_peak_memory_stats()

    ref_width, ref_height, ref_num_frames, ref_fps = read_video_metadata(args.reference_video)

    width = args.width if args.width is not None else round_down_to_multiple(ref_width, 64)
    height = args.height if args.height is not None else round_down_to_multiple(ref_height, 64)
    num_frames = args.num_frames if args.num_frames is not None else ref_num_frames
    frame_rate = args.frame_rate if args.frame_rate is not None else ref_fps

    if width % 64 != 0 or height % 64 != 0:
        raise ValueError("height and width must both be divisible by 64 for the two-stage IC-LoRA pipeline.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not (0.0 <= args.conditioning_attention_strength <= 1.0):
        raise ValueError("conditioning_attention_strength must be between 0.0 and 1.0.")
    source_strength_values = (
        parse_source_strength_values(args.source_strength_values) if args.source_strength_values is not None else None
    )
    source_strength_schedule = build_source_strength_schedule(
        args.source_strength_schedule,
        strength=args.source_strength,
        values=source_strength_values,
        early_strength=args.source_strength_early,
        late_strength=args.source_strength_late,
        fade_start_sigma=args.source_strength_fade_start_sigma,
        fade_end_sigma=args.source_strength_fade_end_sigma,
    )
    stage_1_source_strength_schedule = (
        ConstantSourceStrengthSchedule(1.0) if args.stage_2_branch_mode != "legacy" else source_strength_schedule
    )
    source_strength_log = []
    first_frame_attention_values = (
        parse_first_frame_attention_values(args.first_frame_attention_values)
        if args.first_frame_attention_values is not None
        else None
    )
    first_frame_attention_schedule = build_first_frame_attention_schedule(
        args.first_frame_attention_schedule,
        multiplier=args.first_frame_attention_multiplier,
        values=first_frame_attention_values,
        early_multiplier=args.first_frame_attention_early,
        late_multiplier=args.first_frame_attention_late,
        fade_start_sigma=args.first_frame_attention_fade_start_sigma,
        fade_end_sigma=args.first_frame_attention_fade_end_sigma,
        maximum=args.first_frame_attention_max,
    )
    first_frame_attention_log = []
    if args.stage_2_ic_lora_strength < 0.0:
        raise ValueError("--stage-2-ic-lora-strength must be non-negative.")
    controlled_stage_2 = args.stage_2_branch_mode != "legacy"
    if controlled_stage_2 and args.skip_stage_2:
        raise ValueError("controlled Stage-2 branch modes cannot be combined with --skip-stage-2.")
    if not 0.0 <= args.stage_2_video_mix <= 1.0:
        raise ValueError("--stage-2-video-mix must be between 0.0 and 1.0.")
    if args.stage_2_branch_mode == "spatial":
        if args.stage_2_routing_mask is None:
            raise ValueError("--stage-2-routing-mask is required for spatial routing.")
        if args.stage_2_dress_video_contribution is None or not 0.0 <= args.stage_2_dress_video_contribution <= 1.0:
            raise ValueError("--stage-2-dress-video-contribution must be between 0.0 and 1.0.")
    elif args.stage_2_routing_mask is not None:
        raise ValueError("--stage-2-routing-mask is only valid with --stage-2-branch-mode spatial.")
    if args.stage_2_branch_mode in {"video", "global", "spatial"} and args.stage_2_ic_lora_strength <= 0.0:
        raise ValueError("video-bearing controlled Stage-2 modes require --stage-2-ic-lora-strength > 0.")
    if controlled_stage_2 and args.attention_probe_output:
        raise ValueError("--attention-probe-output is not yet supported with controlled Stage-2 modes.")
    if not 0.0 <= args.stage_2_conditioning_attention_strength <= 1.0:
        raise ValueError("--stage-2-conditioning-attention-strength must be between 0.0 and 1.0.")
    if args.skip_stage_2 and args.stage_2_ic_lora_strength > 0.0:
        raise ValueError("--stage-2-ic-lora-strength has no effect with --skip-stage-2.")
    if args.save_stage_2_input and args.load_stage_2_input:
        raise ValueError("--save-stage-2-input and --load-stage-2-input are mutually exclusive.")
    if args.skip_stage_2 and (args.save_stage_2_input or args.load_stage_2_input):
        raise ValueError("Stage-2 input caching cannot be combined with --skip-stage-2.")
    if args.source_correspondence_bias < 0.0:
        raise ValueError("--source-correspondence-bias must be non-negative.")
    if args.source_correspondence_radius < 0:
        raise ValueError("--source-correspondence-radius must be non-negative.")
    if args.correspondence_mask_feather < 0:
        raise ValueError("--correspondence-mask-feather must be non-negative.")
    if not 0.0 <= args.stage_2_masked_denoise_strength <= 1.0:
        raise ValueError("--stage-2-masked-denoise-strength must be between 0.0 and 1.0.")
    if args.skip_stage_2 and args.stage_2_masked_denoise_strength < 1.0:
        raise ValueError("--stage-2-masked-denoise-strength has no effect with --skip-stage-2.")
    if args.source_correspondence_bias > 0.0 and "rgb" not in args.conditioning_mode:
        raise ValueError("Source correspondence bias requires an RGB conditioning mode.")
    if args.correspondence_mask_mode == "file" and not args.correspondence_mask_file:
        raise ValueError("--correspondence-mask-file is required when mask mode is 'file'.")
    if args.correspondence_mask_mode == "boxes" and not args.correspondence_mask_box:
        raise ValueError("At least one --correspondence-mask-box is required when mask mode is 'boxes'.")
    if args.correspondence_mask_box and any(
        values[0] != int(values[0]) or values[1] != int(values[1]) for values in args.correspondence_mask_box
    ):
        raise ValueError("Correspondence box START and END values must be integers.")
    if args.no_image_conditioning and args.enhance_prompt:
        raise ValueError("--enhance-prompt requires a reference image and cannot be used with --no-image-conditioning.")
    image_conditionings = resolve_image_conditionings(args, num_frames)
    reference_image_replace = resolve_reference_image_replace(args, image_conditionings, num_frames)

    print(
        "Resolved video dimensions: "
        f"reference={ref_width}x{ref_height}, "
        f"pipeline_output={width}x{height}, "
        f"num_frames={num_frames}, "
        f"frame_rate={frame_rate:.3f}"
    )

    lora = LoraPathStrengthAndSDOps(
        path=str(Path(args.ic_lora_path).expanduser().resolve()),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
    stage_2_loras = []
    stage_2_runtime_loras = []
    if args.stage_2_ic_lora_strength > 0.0 or controlled_stage_2:
        stage_2_adapter = LoraPathStrengthAndSDOps(
            path=lora.path,
            strength=args.stage_2_ic_lora_strength,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
        if controlled_stage_2:
            stage_2_runtime_loras.append(stage_2_adapter)
        else:
            stage_2_loras.append(stage_2_adapter)

    video_conditioning = resolve_conditioning_videos(args)

    stage_2_routing_mask = None
    if args.stage_2_routing_mask is not None:
        stage_2_routing_mask = load_file_mask(
            path=args.stage_2_routing_mask,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        print(f"Using preprocessed Stage-2 routing mask: {Path(args.stage_2_routing_mask).expanduser().resolve()}")

    correspondence_mask = None
    if args.source_correspondence_bias > 0.0 or args.stage_2_masked_denoise_strength < 1.0:
        stage_1_height, stage_1_width = height // 2, width // 2
        if args.correspondence_mask_mode == "later-frames":
            correspondence_mask = build_later_frames_mask(
                num_frames=num_frames,
                height=stage_1_height,
                width=stage_1_width,
            )
        elif args.correspondence_mask_mode == "boxes":
            boxes = [
                CorrespondenceMaskBox(int(values[0]), int(values[1]), *values[2:])
                for values in args.correspondence_mask_box
            ]
            correspondence_mask = build_box_mask(
                boxes=boxes,
                num_frames=num_frames,
                height=stage_1_height,
                width=stage_1_width,
                feather=args.correspondence_mask_feather,
            )
        else:
            correspondence_mask = load_file_mask(
                path=args.correspondence_mask_file,
                num_frames=num_frames,
                height=stage_1_height,
                width=stage_1_width,
            )
        print(
            f"Using correspondence mask mode={args.correspondence_mask_mode}, "
            f"bias={args.source_correspondence_bias}, radius={args.source_correspondence_radius}, "
            f"stage_2_masked_denoise_strength={args.stage_2_masked_denoise_strength}"
        )

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=[lora],
        stage_2_loras=stage_2_loras,
        stage_2_runtime_loras=stage_2_runtime_loras,
        device=torch.device("cuda"),
        quantization=args.quantization,
    )

    attention_probe = None
    if args.attention_probe_output:
        attention_probe = AttentionProbe(
            AttentionProbeConfig(
                output_path=Path(args.attention_probe_output).expanduser().resolve(),
                layers=parse_index_set(args.attention_probe_layers),
                steps=parse_index_set(args.attention_probe_steps),
                heads=args.attention_probe_heads,
                query_chunk_size=args.attention_probe_query_chunk_size,
                metadata={
                    "reference_video": str(Path(args.reference_video).expanduser().resolve()),
                    "video_conditioning": [{"path": p, "strength": s} for p, s in video_conditioning],
                    "reference_image": (
                        str(Path(args.reference_image).expanduser().resolve()) if args.reference_image else None
                    ),
                    "output": str(Path(args.output).expanduser().resolve()),
                    "correspondence_mask_mode": args.correspondence_mask_mode,
                    "correspondence_mask_file": args.correspondence_mask_file,
                    "correspondence_mask_boxes": args.correspondence_mask_box,
                    "source_correspondence_bias": args.source_correspondence_bias,
                    "source_correspondence_radius": args.source_correspondence_radius,
                    "conditioning_mode": args.conditioning_mode,
                    "stage_2_masked_denoise_strength": args.stage_2_masked_denoise_strength,
                    "stage_2_ic_lora_strength": args.stage_2_ic_lora_strength,
                    "stage_2_conditioning_attention_strength": args.stage_2_conditioning_attention_strength,
                    "stage_2_branch_mode": args.stage_2_branch_mode,
                    "stage_2_video_mix": args.stage_2_video_mix,
                    "stage_2_routing_mask": args.stage_2_routing_mask,
                    "stage_2_dress_video_contribution": args.stage_2_dress_video_contribution,
                    "stage_2_noise_seed": args.stage_2_noise_seed if args.stage_2_noise_seed is not None else args.seed,
                    "video_strength": args.video_strength,
                    "conditioning_attention_strength": args.conditioning_attention_strength,
                    "source_strength_schedule": args.source_strength_schedule,
                    "source_strength": args.source_strength,
                    "source_strength_values": list(source_strength_values) if source_strength_values else None,
                    "source_strength_early": args.source_strength_early,
                    "source_strength_late": args.source_strength_late,
                    "source_strength_fade_start_sigma": args.source_strength_fade_start_sigma,
                    "source_strength_fade_end_sigma": args.source_strength_fade_end_sigma,
                    "source_strength_routing": args.source_strength_routing,
                    "first_frame_attention_schedule": args.first_frame_attention_schedule,
                    "first_frame_attention_multiplier": args.first_frame_attention_multiplier,
                    "first_frame_attention_values": list(first_frame_attention_values)
                    if first_frame_attention_values
                    else None,
                    "first_frame_attention_early": args.first_frame_attention_early,
                    "first_frame_attention_late": args.first_frame_attention_late,
                    "first_frame_attention_fade_start_sigma": args.first_frame_attention_fade_start_sigma,
                    "first_frame_attention_fade_end_sigma": args.first_frame_attention_fade_end_sigma,
                    "first_frame_attention_max": args.first_frame_attention_max,
                    "no_image_conditioning": args.no_image_conditioning,
                    "image_conditions": [image._asdict() for image in image_conditionings],
                    "reference_image_replace": sorted(reference_image_replace)
                    if reference_image_replace is not None
                    else None,
                    "skip_stage_2": args.skip_stage_2,
                    "seed": args.seed,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "frame_rate": frame_rate,
                },
            )
        )
        print(f"Writing attention probe metrics to {attention_probe.output_path}")

    initial_video_latent = None
    if args.initial_video_latent:
        latent_path = Path(args.initial_video_latent).expanduser().resolve()
        payload = torch.load(latent_path, map_location="cpu")
        initial_video_latent = payload.get("latent", payload) if isinstance(payload, dict) else payload
        if not isinstance(initial_video_latent, torch.Tensor):
            raise TypeError(f"Expected a tensor at {latent_path}, got {type(initial_video_latent)!r}")
        print(f"Loaded initial Stage 1 latent from {latent_path} with shape {tuple(initial_video_latent.shape)}")

    resolved_prompt = build_prompt(args.prompt, args.negative_prompt)
    stage_2_cache_metadata = build_stage_2_cache_metadata(
        args,
        prompt=resolved_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        image_conditionings=image_conditionings,
        video_conditioning=video_conditioning,
    )
    cached_stage_2_video_latent = None
    cached_stage_2_audio_latent = None
    cache_manifest = None
    if args.load_stage_2_input:
        cached_stage_2_video_latent, cached_stage_2_audio_latent, cache_manifest = load_stage2_input_cache(
            args.load_stage_2_input,
            expected_metadata=stage_2_cache_metadata,
        )
        print(
            "Loaded verified Stage-2 input cache "
            f"video_checksum={cache_manifest['video_checksum']} "
            f"audio_checksum={cache_manifest['audio_checksum']}"
        )

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=resolved_prompt,
        seed=args.seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=image_conditionings,
        video_conditioning=video_conditioning,
        correspondence_mask=correspondence_mask,
        source_correspondence_bias=args.source_correspondence_bias,
        source_correspondence_radius=args.source_correspondence_radius,
        enhance_prompt=args.enhance_prompt,
        stage_2_masked_denoise_strength=args.stage_2_masked_denoise_strength,
        stage_2_conditioning_attention_strength=args.stage_2_conditioning_attention_strength,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        attention_probe=attention_probe,
        reference_image_replace=reference_image_replace,
        initial_video_latent=initial_video_latent,
        source_strength_schedule=stage_1_source_strength_schedule,
        stage_2_source_strength_schedule=source_strength_schedule,
        source_strength_routing=SourceStrengthRouting(args.source_strength_routing),
        source_strength_log=source_strength_log,
        first_frame_attention_schedule=first_frame_attention_schedule,
        first_frame_attention_log=first_frame_attention_log,
        stage_2_branch_mode=args.stage_2_branch_mode,
        stage_2_video_mix=args.stage_2_video_mix,
        stage_2_routing_mask=stage_2_routing_mask,
        stage_2_dress_video_contribution=args.stage_2_dress_video_contribution,
        stage_2_noise_seed=args.stage_2_noise_seed,
        stage_2_prediction_dir=args.stage_2_prediction_dir,
        cached_stage_2_video_latent=cached_stage_2_video_latent,
        cached_stage_2_audio_latent=cached_stage_2_audio_latent,
        save_stage_2_input_path=args.save_stage_2_input,
        stage_2_input_metadata=stage_2_cache_metadata,
        # streaming_prefetch_count=args.streaming_prefetch_count,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=video,
        fps=int(round(frame_rate)),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=output_chunks,
    )

    if controlled_stage_2:
        if cache_manifest is None and args.save_stage_2_input:
            cache_path = Path(args.save_stage_2_input).expanduser().resolve()
            cache_manifest = json.loads(cache_path.with_suffix(cache_path.suffix + ".json").read_text())
        run_manifest = {
            "stage_2_branch_mode": args.stage_2_branch_mode,
            "stage_2_video_mix": args.stage_2_video_mix,
            "stage_2_routing_mask": (
                str(Path(args.stage_2_routing_mask).expanduser().resolve()) if args.stage_2_routing_mask else None
            ),
            "stage_2_dress_video_contribution": args.stage_2_dress_video_contribution,
            "stage_2_ic_lora_strength": args.stage_2_ic_lora_strength,
            "stage_2_noise_seed": args.stage_2_noise_seed if args.stage_2_noise_seed is not None else args.seed,
            "stage_2_prediction_dir": args.stage_2_prediction_dir,
            "stage_2_input_cache": args.load_stage_2_input or args.save_stage_2_input,
            "stage_2_input_video_checksum": cache_manifest.get("video_checksum") if cache_manifest else None,
            "stage_2_input_audio_checksum": cache_manifest.get("audio_checksum") if cache_manifest else None,
            "stage_1_fingerprint": stage_2_cache_metadata["stage_1_fingerprint"],
            "output": str(output_path),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        run_manifest_path = output_path.with_suffix(output_path.suffix + ".stage2.json")
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote controlled Stage-2 run manifest to {run_manifest_path}")

    if args.source_strength_log:
        log_path = Path(args.source_strength_log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "evaluation_index": row.evaluation_index,
                "nominal_step_index": row.nominal_step_index,
                "sigma": row.sigma,
                "next_sigma": row.next_sigma,
                "progress": row.progress,
                "g_source": row.g_source,
                "routing": row.routing.value,
                "target_token_count": row.target_token_count,
                "source_token_ranges": row.source_token_ranges,
                "composed_existing_mask": row.composed_existing_mask,
                "num_evaluations": row.num_evaluations,
            }
            for row in source_strength_log
        ]
        log_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote source strength log to {log_path}")

    if args.first_frame_attention_log:
        log_path = Path(args.first_frame_attention_log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "evaluation_index": row.evaluation_index,
                "nominal_step_index": row.nominal_step_index,
                "sigma": row.sigma,
                "next_sigma": row.next_sigma,
                "progress": row.progress,
                "h_first_frame": row.h_first_frame,
                "target_token_count": row.target_token_count,
                "frame0_token_range": row.frame0_token_range,
                "later_target_token_range": row.later_target_token_range,
                "composed_existing_mask": row.composed_existing_mask,
                "num_evaluations": row.num_evaluations,
            }
            for row in first_frame_attention_log
        ]
        log_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote first-frame attention log to {log_path}")

    print(f"Saved video to {output_path}")
    print(
        "Generation settings: "
        f"{width}x{height}, {num_frames} frames at {frame_rate:.3f} fps, "
        f"image conditions={len(image_conditionings)}, "
        f"video conditions={[(p, s) for p, s in video_conditioning]}, "
        f"conditioning mode={args.conditioning_mode}, "
        f"source strength schedule={args.source_strength_schedule}, "
        f"source strength routing={args.source_strength_routing}, "
        f"first-frame attention schedule={args.first_frame_attention_schedule}, "
        f"Stage 2 IC-LoRA strength={args.stage_2_ic_lora_strength}, "
        f"Stage 2 conditioning attention strength={args.stage_2_conditioning_attention_strength}, "
        f"no image conditioning={args.no_image_conditioning}, "
        f"reference image replace="
        f"{sorted(reference_image_replace) if reference_image_replace is not None else 'default(0)'}"
    )
    if args.no_image_conditioning and not args.skip_stage_2:
        print(
            "Warning: --no-image-conditioning leaves Stage 2 without an image anchor; "
            "--skip-stage-2 is usually the safer choice for minimum drift tests."
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
