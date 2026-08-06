#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from style_transfer_preprocess import (
    PreparedStyleTransferInputs,
    add_preprocess_args,
    build_source_video,
    prepare_style_transfer_inputs,
    resolve_control_videos,
    resolve_source_frames,
)

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
IC_LORA_STYLE_TRANSFER_SCRIPT = SCRIPT_ROOT / "style_transfer_ltx23" / "run_ic_lora_style_transfer.py"


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
    control_videos: list[str],
    reference_image: str | None,
    image_conditions: list[tuple[str, int] | tuple[str, int, float, int]] | None = None,
    num_frames: int | None = None,
    output: str | None = None,
    reference_image_replace: list[int] | None = None,
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
        control_videos[0],
        "--output",
        output or args.output,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--seed",
        str(args.seed),
        "--video-strength",
        str(args.video_strength),
        "--conditioning-mode",
        args.conditioning_mode,
        "--conditioning-attention-strength",
        str(args.conditioning_attention_strength),
        "--stage-2-ic-lora-strength",
        str(args.stage_2_ic_lora_strength),
        "--stage-2-conditioning-attention-strength",
        str(args.stage_2_conditioning_attention_strength),
        "--correspondence-mask-mode",
        args.correspondence_mask_mode,
        "--correspondence-mask-feather",
        str(args.correspondence_mask_feather),
        "--source-correspondence-bias",
        str(args.source_correspondence_bias),
        "--source-correspondence-radius",
        str(args.source_correspondence_radius),
        "--stage-2-masked-denoise-strength",
        str(args.stage_2_masked_denoise_strength),
    ]

    if args.correspondence_mask_file:
        command.extend(["--correspondence-mask-file", args.correspondence_mask_file])
    for box in args.correspondence_mask_box or []:
        command.extend(["--correspondence-mask-box", *(str(value) for value in box)])

    if args.no_image_conditioning:
        command.append("--no-image-conditioning")
    else:
        if reference_image is None:
            raise ValueError("reference_image is required unless --no-image-conditioning is used.")
        command.extend(
            [
                "--reference-image",
                reference_image,
                "--image-frame-idx",
                str(args.image_frame_idx),
                "--image-strength",
                str(args.image_strength),
                "--image-crf",
                str(args.image_crf),
            ]
        )

    if args.conditioning_mode in ("rgb+depth", "edge+depth+rgb") and len(control_videos) > 1:
        command.extend(["--depth-conditioning-video", control_videos[1]])
    if args.conditioning_mode == "edge+rgb" and len(control_videos) > 1:
        command.extend(["--edge-conditioning-video", control_videos[1]])
    if args.conditioning_mode == "edge+depth+rgb" and len(control_videos) > 2:
        command.extend(["--edge-conditioning-video", control_videos[2]])

    if getattr(args, "rgb_strength", None) is not None:
        command.extend(["--rgb-strength", str(args.rgb_strength)])
    if getattr(args, "depth_strength", None) is not None:
        command.extend(["--depth-strength", str(args.depth_strength)])
    if getattr(args, "edge_strength", None) is not None:
        command.extend(["--edge-strength", str(args.edge_strength)])

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
        for condition in image_conditions:
            image_path, frame_idx = condition[0], condition[1]
            strength = condition[2] if len(condition) > 2 else args.image_strength
            crf = condition[3] if len(condition) > 3 else args.image_crf
            command.extend(
                [
                    "--image-condition",
                    image_path,
                    str(frame_idx),
                    str(strength),
                    str(crf),
                ]
            )

    return command


def prepare_style_transfer_inputs_without_reference_image(args: argparse.Namespace) -> PreparedStyleTransferInputs:
    input_path = Path(args.input_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    source_video = build_source_video(args, input_path, work_dir)
    reference_frame_indices, source_reference_frames = resolve_source_frames(args, source_video, work_dir)
    source_first_frame = source_reference_frames[0]
    control_videos = resolve_control_videos(args, source_video, input_path, work_dir)

    return PreparedStyleTransferInputs(
        source_video=str(source_video),
        source_first_frame=str(source_first_frame),
        reference_image="",
        source_reference_frames=[str(path) for path in source_reference_frames],
        reference_images=[],
        reference_frame_indices=reference_frame_indices,
        control_videos=[str(v) for v in control_videos],
        work_dir=str(work_dir),
    )


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
    parser.add_argument("--no-image-conditioning", action="store_true")
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
    parser.add_argument(
        "--rgb-strength",
        type=float,
        default=None,
        help="Strength for the RGB video condition in rgb+depth mode. Defaults to --video-strength.",
    )
    parser.add_argument(
        "--correspondence-mask-mode",
        choices=("later-frames", "boxes", "file"),
        default="later-frames",
    )
    parser.add_argument("--correspondence-mask-file", default=None)
    parser.add_argument(
        "--correspondence-mask-box",
        type=float,
        nargs=6,
        action="append",
        default=None,
        metavar=("START", "END", "X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--correspondence-mask-feather", type=int, default=0)
    parser.add_argument("--source-correspondence-bias", type=float, default=0.0)
    parser.add_argument("--source-correspondence-radius", type=int, default=0)
    parser.add_argument(
        "--stage-2-masked-denoise-strength",
        "--stage-2-masked-noise-strength",
        type=float,
        default=1.0,
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
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--stage-2-ic-lora-strength", type=float, default=0.0)
    parser.add_argument("--stage-2-conditioning-attention-strength", type=float, default=1.0)
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
        "--image-condition",
        nargs=4,
        action="append",
        default=None,
        metavar=("IMAGE_PATH", "FRAME_IDX", "STRENGTH", "CRF"),
        help=(
            "Override image conditions passed to IC-LoRA, bypassing the preprocessing stage. "
            "Format: IMAGE_PATH FRAME_IDX STRENGTH CRF. Can be repeated for multiple conditions."
        ),
    )
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

    if args.no_image_conditioning:
        if args.image_condition:
            raise ValueError("--no-image-conditioning cannot be combined with --image-condition.")
        if args.enhance_prompt:
            raise ValueError("--enhance-prompt cannot be combined with --no-image-conditioning.")
        if args.reference_image_replace not in (None, []):
            raise ValueError(
                "--reference-image-replace requires image conditions and cannot be used with --no-image-conditioning."
            )

    if args.no_image_conditioning:
        prepared = prepare_style_transfer_inputs_without_reference_image(args)
    else:
        prepared = prepare_style_transfer_inputs(args)
    if args.print_prepared_inputs:
        print(f"Prepared inputs: {asdict(prepared)}")

    if args.no_image_conditioning:
        image_conditions = None
    elif args.image_condition:
        image_conditions = [
            (path, int(frame_idx), float(strength), int(crf)) for path, frame_idx, strength, crf in args.image_condition
        ]
    else:
        image_conditions = list(zip(prepared.reference_images, prepared.reference_frame_indices, strict=True))
    if args.reference_image_replace is not None and image_conditions is not None:
        condition_frame_indices = {int(c[1]) for c in image_conditions}
        missing = [idx for idx in sorted(set(args.reference_image_replace)) if idx not in condition_frame_indices]
        if missing:
            raise ValueError(
                f"--reference-image-replace can only name frames that have image conditions. Missing: {missing}"
            )
    command = build_style_transfer_command(
        args,
        prepared.source_video,
        prepared.control_videos,
        prepared.reference_image if not args.no_image_conditioning else None,
        image_conditions=image_conditions,
        reference_image_replace=args.reference_image_replace,
    )
    run_command(command)

    print("Single-sequence translation completed successfully.")
    print(f"Source video: {prepared.source_video}")
    print(f"Source first frame: {prepared.source_first_frame}")
    print(f"Reference images: {prepared.reference_images}")
    print(f"Reference frame indices: {prepared.reference_frame_indices}")
    print(
        "Reference image replace: "
        f"{args.reference_image_replace if args.reference_image_replace is not None else 'default(0)'}"
    )
    print(f"Control videos: {prepared.control_videos}")
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()
