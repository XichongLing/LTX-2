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
from ltx_pipelines.utils.args import ImageConditioningInput, QuantizationAction, QUANTIZATION_POLICIES
from ltx_pipelines.utils.media_io import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
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
    parser.add_argument("--reference-image", required=True, help="Reference image whose style/appearance should influence the output.")
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
        choices=("rgb", "depth", "edge"),
        default="rgb",
        help="Interpretation of the conditioning video expected by the IC-LoRA.",
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


def validate_conditioning_video_path(args: argparse.Namespace) -> str:
    conditioning_path = Path(args.conditioning_video).expanduser().resolve()
    if not is_readable_video_file(conditioning_path):
        raise ValueError(
            f"conditioning video does not exist or is unreadable: {conditioning_path}. "
            "Prepare it with style_transfer_preprocess.py or translate_single_sequence.py."
        )
    print(f"Using conditioning video: {conditioning_path}")
    return str(conditioning_path)


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
    if args.image_frame_idx < 0 or args.image_frame_idx >= num_frames:
        raise ValueError(f"image-frame-idx must be in [0, {num_frames - 1}]")

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

    conditioning_video_path = validate_conditioning_video_path(args)

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=[lora],
        device=torch.device("cuda"),
        quantization=args.quantization,
    )

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=build_prompt(args.prompt, args.negative_prompt),
        seed=args.seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=[
            ImageConditioningInput(
                path=str(Path(args.reference_image).expanduser().resolve()),
                frame_idx=args.image_frame_idx,
                strength=args.image_strength,
                crf=args.image_crf,
            )
        ],
        video_conditioning=[
            (
                conditioning_video_path,
                args.video_strength,
            )
        ],
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
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
        f"image strength={args.image_strength}, video strength={args.video_strength}, "
        f"conditioning mode={args.conditioning_mode}"
    )

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
