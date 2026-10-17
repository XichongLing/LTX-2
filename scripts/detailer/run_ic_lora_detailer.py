from __future__ import annotations

import argparse
import logging
import sys
from fractions import Fraction
from pathlib import Path

import av
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC_DIRS = (
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
)
for src_dir in PACKAGE_SRC_DIRS:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: E402
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: E402
from ltx_pipelines import ICLoraPipeline  # noqa: E402
from ltx_pipelines.utils.args import QUANTIZATION_POLICIES, ImageConditioningInput, QuantizationAction  # noqa: E402
from ltx_pipelines.utils.attention_probe import AttentionProbe, AttentionProbeConfig, parse_index_set  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402


class ImageConditioningAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (3, 4):
            raise argparse.ArgumentError(
                self,
                f"{option_string} requires PATH FRAME_IDX STRENGTH [CRF], got {len(values)} values",
            )
        image = ImageConditioningInput(
            path=str(Path(values[0]).expanduser().resolve()),
            frame_idx=int(values[1]),
            strength=float(values[2]),
            crf=int(values[3]) if len(values) == 4 else 33,
        )
        images = getattr(namespace, self.dest) or []
        images.append(image)
        setattr(namespace, self.dest, images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an LTX-2 IC-LoRA detailer pass. The input video is used as RGB "
            "video conditioning, and ltx-2-19b-ic-lora-detailer.safetensors is "
            "loaded as the IC-LoRA adapter."
        )
    )
    parser.add_argument("--distilled-checkpoint-path", required=True, help="Path to the distilled LTX-2 checkpoint.")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Path to the x2 spatial upsampler checkpoint.")
    parser.add_argument("--gemma-root", required=True, help="Path to the Gemma text-encoder directory.")
    parser.add_argument(
        "--detailer-lora-path",
        required=True,
        help="Path to ltx-2-19b-ic-lora-detailer.safetensors.",
    )
    parser.add_argument(
        "--input-video",
        required=True,
        help="Video to detail/refine; used as RGB IC-LoRA conditioning.",
    )
    parser.add_argument(
        "--conditioning-video",
        default=None,
        help=(
            "Optional separate conditioning video. Defaults to --input-video. "
            "Use this when the detail pass should follow a preprocessed RGB/depth/edge control video."
        ),
    )
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--prompt",
        default=(
            "Refine the input video while preserving the same composition, camera motion, timing, subject identity, "
            "and scene layout. Add crisp high-frequency detail, clean edges, coherent texture, natural lighting, "
            "and stable temporally consistent surfaces."
        ),
        help="Prompt describing the refined output.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "flicker, temporal jitter, warped face, distorted hands, extra fingers, mushy texture, "
            "oversharpened halos, "
            "plastic skin, noisy grain, blocky compression artifacts, changed identity, changed composition"
        ),
        help="Negative prompt appended to the detailer prompt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. Defaults to input height rounded down to a multiple of 64.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. Defaults to input width rounded down to a multiple of 64.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of output frames. Defaults to input frame count.",
    )
    parser.add_argument("--frame-rate", type=float, default=None, help="Output FPS. Defaults to input FPS.")
    parser.add_argument(
        "--video-strength",
        type=float,
        default=1.0,
        help=(
            "Strength for the input/conditioning video latent. "
            "Higher values preserve the control video more strongly."
        ),
    )
    parser.add_argument(
        "--detailer-lora-strength",
        type=float,
        default=1.0,
        help="Strength for the detailer IC-LoRA adapter.",
    )
    parser.add_argument(
        "--conditioning-attention-strength",
        type=float,
        default=1.0,
        help="Attention strength for IC-LoRA video conditioning, in [0, 1].",
    )
    parser.add_argument(
        "--first-frame-attention-multiplier",
        type=float,
        default=1.0,
        help=(
            "Attention multiplier for future target tokens attending to in-place frame-0 replacement tokens. "
            "1.0 preserves current behavior; values >1.0 add positive attention-logit bias."
        ),
    )
    parser.add_argument(
        "--image-condition",
        action=ImageConditioningAction,
        nargs="+",
        default=None,
        metavar="IMAGE_CONDITION",
        help=(
            "Optional image anchor: PATH FRAME_IDX STRENGTH [CRF]. Can be repeated. "
            "Useful for pinning the first frame or a face/identity reference."
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
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Skip the spatial upsampling/refinement stage for faster half-resolution previews.",
    )
    parser.add_argument("--enhance-prompt", action="store_true", help="Let Gemma enhance the prompt.")
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
            "Example: --quantization fp8-cast"
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
        fps_value = video_stream.average_rate or video_stream.base_rate or Fraction(24, 1)
        fps = float(fps_value)

        frames = video_stream.frames
        if not frames or frames <= 0:
            frames = 0
            for packet in container.demux(video_stream):
                for _frame in packet.decode():
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


def validate_args(args: argparse.Namespace, num_frames: int) -> None:
    if args.width % 64 != 0 or args.height % 64 != 0:
        raise ValueError("height and width must both be divisible by 64.")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not (0.0 <= args.conditioning_attention_strength <= 1.0):
        raise ValueError("conditioning_attention_strength must be between 0.0 and 1.0.")
    if args.detailer_lora_strength < 0.0:
        raise ValueError("detailer_lora_strength must be non-negative.")
    if args.video_strength < 0.0:
        raise ValueError("video_strength must be non-negative.")
    if args.first_frame_attention_multiplier <= 0.0:
        raise ValueError("first_frame_attention_multiplier must be greater than 0.0.")

    for image in args.image_condition or []:
        if image.frame_idx < 0 or image.frame_idx >= num_frames:
            raise ValueError(f"image condition frame_idx must be in [0, {num_frames - 1}]: {image}")

    if args.reference_image_replace is not None:
        replace_frame_indices = sorted(set(args.reference_image_replace))
        out_of_range = [idx for idx in replace_frame_indices if idx < 0 or idx >= num_frames]
        if out_of_range:
            raise ValueError(
                f"reference-image-replace frame index must be in [0, {num_frames - 1}]: {out_of_range}"
            )

        image_frame_indices = {image.frame_idx for image in args.image_condition or []}
        missing = [idx for idx in replace_frame_indices if idx not in image_frame_indices]
        if missing:
            raise ValueError(
                "--reference-image-replace can only name frames that have image conditions. "
                f"Missing image conditions for frames: {missing}."
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.quiet_layer_streaming:
        logging.getLogger("ltx_core.layer_streaming").setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run LTX-2 IC-LoRA detailer inference.")

    input_video = Path(args.input_video).expanduser().resolve()
    conditioning_video = Path(args.conditioning_video or args.input_video).expanduser().resolve()
    in_width, in_height, in_frames, in_fps = read_video_metadata(str(input_video))

    args.width = args.width if args.width is not None else round_down_to_multiple(in_width, 64)
    args.height = args.height if args.height is not None else round_down_to_multiple(in_height, 64)
    args.num_frames = args.num_frames if args.num_frames is not None else in_frames
    args.frame_rate = args.frame_rate if args.frame_rate is not None else in_fps
    validate_args(args, args.num_frames)

    logging.info(
        "Resolved detailer dimensions: "
        f"input={in_width}x{in_height}, output={args.width}x{args.height}, "
        f"frames={args.num_frames}, fps={args.frame_rate:.3f}"
    )
    logging.info("Using IC-LoRA conditioning video: %s", conditioning_video)

    lora = LoraPathStrengthAndSDOps(
        path=str(Path(args.detailer_lora_path).expanduser().resolve()),
        strength=args.detailer_lora_strength,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )

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
                    "input_video": str(input_video),
                    "conditioning_video": str(conditioning_video),
                    "output": str(Path(args.output).expanduser().resolve()),
                    "video_strength": args.video_strength,
                    "conditioning_attention_strength": args.conditioning_attention_strength,
                    "first_frame_attention_multiplier": args.first_frame_attention_multiplier,
                    "image_conditions": [image._asdict() for image in args.image_condition or []],
                    "reference_image_replace": sorted(set(args.reference_image_replace))
                    if args.reference_image_replace is not None
                    else None,
                    "skip_stage_2": args.skip_stage_2,
                    "seed": args.seed,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "frame_rate": args.frame_rate,
                },
            )
        )
        logging.info("Writing attention probe metrics to %s", attention_probe.output_path)

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=build_prompt(args.prompt, args.negative_prompt),
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.image_condition or [],
        video_conditioning=[(str(conditioning_video), args.video_strength)],
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        attention_probe=attention_probe,
        reference_image_replace=set(args.reference_image_replace) if args.reference_image_replace is not None else None,
        first_frame_attention_multiplier=args.first_frame_attention_multiplier,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks = get_video_chunks_number(args.num_frames, tiling_config)
    encode_video(
        video=video,
        fps=round(args.frame_rate),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=output_chunks,
    )

    logging.info("Saved detailed video to %s", output_path)
    logging.info(
        "Detailer settings: "
        f"detailer_lora_strength={args.detailer_lora_strength}, "
        f"video_strength={args.video_strength}, "
        f"conditioning_attention_strength={args.conditioning_attention_strength}, "
        f"first_frame_attention_multiplier={args.first_frame_attention_multiplier}, "
        f"image_conditions={len(args.image_condition or [])}, "
        f"reference_image_replace={sorted(set(args.reference_image_replace)) if args.reference_image_replace is not None else 'default(0)'}, "
        f"skip_stage_2={args.skip_stage_2}"
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
