#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
IC_LORA_EDIT_SCRIPT = SCRIPT_ROOT / "local_edit" / "run_edit_anything.py"


DEFAULT_PROMPT = (
    "Follow the motion, timing, camera movement, and scene layout of the reference video, "
    "but render the result in the appearance, materials, lighting, and style implied by the reference image."
)


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def build_edit_command(
    args: argparse.Namespace,
    source_video: str,
    reference_video: str | None,
    reference_image: str,
    image_conditions: list[tuple[str, int]] | None = None,
    num_frames: int | None = None,
    output: str | None = None,
    reference_image_replace: list[int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(IC_LORA_EDIT_SCRIPT),
        "--distilled-checkpoint-path",
        args.distilled_checkpoint_path,
        "--spatial-upsampler-path",
        args.spatial_upsampler_path,
        "--gemma-root",
        args.gemma_root,
        "--ic-lora-path",
        args.ic_lora_path,
        "--source-video",
        source_video,
        "--reference-video",
        source_video,
        "--reference-image",
        reference_image,
        "--output",
        output or args.output,
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
        "--conditioning-attention-strength",
        str(args.conditioning_attention_strength),
    ]

    if args.width is not None:
        command.extend(["--width", str(args.width)])
    if args.height is not None:
        command.extend(["--height", str(args.height)])
    resolved_num_frames = num_frames if num_frames is not None else args.num_frames
    if resolved_num_frames is not None:
        command.extend(["--num-frames", str(resolved_num_frames)])
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
    if args.attention_probe_output:
        command.extend(["--attention-probe-output", args.attention_probe_output])
        command.extend(["--attention-probe-layers", args.attention_probe_layers])
        command.extend(["--attention-probe-steps", args.attention_probe_steps])
        command.extend(["--attention-probe-heads", args.attention_probe_heads])
        command.extend(["--attention-probe-query-chunk-size", str(args.attention_probe_query_chunk_size)])
    if args.quantization:
        command.extend(["--quantization", *args.quantization])

    replace_frame_indices = (
        reference_image_replace
        if reference_image_replace is not None
        else getattr(args, "reference_image_replace", None)
    )
    if replace_frame_indices is not None:
        command.append("--reference-image-replace")
        command.extend(str(frame_idx) for frame_idx in replace_frame_indices)

    if image_conditions:
        for image_path, frame_idx in image_conditions:
            command.extend(
                [
                    "--image-condition",
                    image_path,
                    str(frame_idx),
                    str(args.image_strength),
                    str(args.image_crf),
                ]
            )

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit the source videos."
    )
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
    parser.add_argument(
        "--reference-image-replace",
        type=int,
        nargs="*",
        default=None,
        metavar="FRAME_IDX",
        help=(
            "Reference-frame indices to inject as in-place latent replacements. "
            "When omitted, only frame 0 uses the existing replacement behavior."
        ),
    )
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--quiet-layer-streaming", action="store_true")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None)
    parser.add_argument("--attention-probe-output", default=None)
    parser.add_argument("--attention-probe-layers", default="0,8,16,24,32,40,47")
    parser.add_argument("--attention-probe-steps", default="all")
    parser.add_argument("--attention-probe-heads", choices=("mean", "all"), default="mean")
    parser.add_argument("--attention-probe-query-chunk-size", type=int, default=128)
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

    if args.reference_image:
        image_conditions = [(args.reference_image, args.image_frame_idx)]
    else:
        image_conditions = None
    if args.reference_image_replace is not None:
        prepared_frame_indices = set(frame_idx for _, frame_idx in args.reference_image_replace)
        missing = [idx for idx in sorted(set(args.reference_image_replace)) if idx not in prepared_frame_indices]
        if missing:
            raise ValueError(
                "--reference-image-replace can only name prepared reference frames. "
                f"Missing from --reference-frame-list: {missing}"
            )
    command = build_edit_command(
        args,
        args.source_video,
        args.reference_image,
        image_conditions=image_conditions,
        num_frames=args.num_frames,
        output=output_path,
        reference_image_replace=args.reference_image_replace,
    )
    run_command(command)

    print("Single-sequence translation completed successfully.")
    print(f"Source video: {args.source_video}")
    print(f"Source first frame: {args.source_first_frame}")
    print(f"Reference images: {args.reference_images}")
    print(
        "Reference image replace: "
        f"{args.reference_image_replace if args.reference_image_replace is not None else 'default(0)'}"
    )
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()
