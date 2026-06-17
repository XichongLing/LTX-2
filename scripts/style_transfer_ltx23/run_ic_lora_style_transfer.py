#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines import ICLoraPipeline
from ltx_pipelines.utils.attention_probe import AttentionProbe, AttentionProbeConfig, parse_index_set
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    QuantizationAction,
    QUANTIZATION_POLICIES,
    _PipelineArgumentParser,
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
    parser.add_argument("--reference-video", required=True, help="Reference video whose motion and camera should be followed.")
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
        required=True,
        help="Reference image whose style/appearance should influence the output.",
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
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt text appended to the main prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output height. Defaults to the reference video height rounded down to a multiple of 64.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Defaults to the reference video width rounded down to a multiple of 64.")
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
    parser.add_argument("--image-strength", type=float, default=1.0, help="Strength for the reference image conditioning.")
    parser.add_argument("--image-crf", type=int, default=33, help="CRF-like preprocessing parameter for image conditioning.")
    parser.add_argument("--video-strength", type=float, default=1.0, help="Strength for the reference video IC-LoRA conditioning.")
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
        raise ValueError(
            f"reference-image-replace frame index must be in [0, {num_frames - 1}]: {out_of_range}"
        )

    image_frame_indices = {conditioning.frame_idx for conditioning in image_conditionings}
    missing = [idx for idx in replace_frame_indices if idx not in image_frame_indices]
    if missing:
        raise ValueError(
            "--reference-image-replace can only name frames that have image conditions. "
            f"Missing image conditions for frames: {missing}. "
            "Add matching --image-condition entries or include the frames in --reference-frame-list."
        )

    return set(replace_frame_indices)


def main() -> None:
    args = parse_args()

    if args.quiet_layer_streaming:
        logging.getLogger("ltx_core.layer_streaming").setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run LTX-2 IC-LoRA inference.")

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

    video_conditioning = resolve_conditioning_videos(args)

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=[lora],
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
                    "reference_image": str(Path(args.reference_image).expanduser().resolve()),
                    "output": str(Path(args.output).expanduser().resolve()),
                    "conditioning_mode": args.conditioning_mode,
                    "video_strength": args.video_strength,
                    "conditioning_attention_strength": args.conditioning_attention_strength,
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

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=build_prompt(args.prompt, args.negative_prompt),
        seed=args.seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=image_conditionings,
        video_conditioning=video_conditioning,
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        attention_probe=attention_probe,
        reference_image_replace=reference_image_replace,
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

    print(f"Saved video to {output_path}")
    print(
        "Generation settings: "
        f"{width}x{height}, {num_frames} frames at {frame_rate:.3f} fps, "
        f"image conditions={len(image_conditionings)}, "
        f"video conditions={[(p, s) for p, s in video_conditioning]}, "
        f"conditioning mode={args.conditioning_mode}, "
        f"reference image replace="
        f"{sorted(reference_image_replace) if reference_image_replace is not None else 'default(0)'}"
    )

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
