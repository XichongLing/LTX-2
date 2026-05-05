#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from style_transfer_preprocess import add_preprocess_args, prepare_style_transfer_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
IC_LORA_STYLE_TRANSFER_SCRIPT = REPO_ROOT / "scripts" / "run_ic_lora_style_transfer.py"


DEFAULT_PROMPT = (
    "Follow the motion, timing, camera movement, and scene layout of the reference video, "
    "but render the result in the appearance, materials, lighting, and style implied by the reference image."
)


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def build_style_transfer_command(
    args: argparse.Namespace,
    source_video: str,
    control_video: str,
    reference_image: str,
) -> list[str]:
    command = [
        sys.executable,
        str(IC_LORA_STYLE_TRANSFER_SCRIPT),
        "--distilled-checkpoint-path",
        args.distilled_checkpoint_path,
        "--spatial-upsampler-path",
        args.spatial_upsampler_path,
        "--gemma-root",
        args.gemma_root,
        "--ic-lora-path",
        args.ic_lora_path,
        "--reference-video",
        source_video,
        "--conditioning-video",
        control_video,
        "--reference-image",
        reference_image,
        "--output",
        args.output,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--seed",
        str(args.seed),
        "--image-frame-idx",
        str(args.image_frame_idx),
        "--image-strength",
        str(args.image_strength),
        "--image-crf",
        str(args.image_crf),
        "--video-strength",
        str(args.video_strength),
        "--conditioning-mode",
        args.conditioning_mode,
        "--conditioning-attention-strength",
        str(args.conditioning_attention_strength),
    ]

    if args.width is not None:
        command.extend(["--width", str(args.width)])
    if args.height is not None:
        command.extend(["--height", str(args.height)])
    if args.num_frames is not None:
        command.extend(["--num-frames", str(args.num_frames)])
    if args.frame_rate is not None:
        command.extend(["--frame-rate", str(args.frame_rate)])
    if args.skip_stage_2:
        command.append("--skip-stage-2")
    if args.enhance_prompt:
        command.append("--enhance-prompt")
    if args.quiet_layer_streaming:
        command.append("--quiet-layer-streaming")
    if args.streaming_prefetch_count is not None:
        command.extend(["--streaming-prefetch-count", str(args.streaming_prefetch_count)])
    if args.quantization:
        command.extend(["--quantization", *args.quantization])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate one source sequence/video by preprocessing inputs, then running IC-LoRA style transfer."
    )
    add_preprocess_args(parser)
    parser.set_defaults(prompt=DEFAULT_PROMPT)
    parser.add_argument("--output", required=True, help="Final stylized MP4 output path.")
    parser.add_argument("--distilled-checkpoint-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--ic-lora-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--image-frame-idx", type=int, default=0)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--quiet-layer-streaming", action="store_true")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None)
    parser.add_argument("--quantization", nargs="+", default=None)
    parser.add_argument(
        "--print-prepared-inputs",
        action="store_true",
        help="Print the prepared artifact paths before running IC-LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)

    prepared = prepare_style_transfer_inputs(args)
    if args.print_prepared_inputs:
        print(f"Prepared inputs: {asdict(prepared)}")

    command = build_style_transfer_command(args, prepared.source_video, prepared.control_video, prepared.reference_image)
    run_command(command)

    print("Single-sequence translation completed successfully.")
    print(f"Source video: {prepared.source_video}")
    print(f"Source first frame: {prepared.source_first_frame}")
    print(f"Reference image: {prepared.reference_image}")
    print(f"Control video: {prepared.control_video}")
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()
