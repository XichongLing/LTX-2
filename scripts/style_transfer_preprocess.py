#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SEQUENCE_TO_VIDEO_SCRIPT = SCRIPTS_DIR / "sequence_to_video.py"
DEFAULT_FLUX_RUNNER = REPO_ROOT.parent / "flux.2_dev" / "run_flux_first_frame_style_transfer.py"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class PreparedStyleTransferInputs:
    source_video: str
    source_first_frame: str
    reference_image: str
    control_video: str
    work_dir: str


def natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def resolve_frame_pattern(dataset: str, input_path: Path, pattern: str) -> str:
    if pattern != "auto":
        return pattern
    if dataset == "mpi-sintel":
        return "frame_*.png"
    if dataset == "synfmc":
        return f"{input_path.name}_*.png"
    if dataset == "vkitti":
        return "*.png"
    return "*"


def collect_frames(input_path: Path, pattern: str) -> list[Path]:
    frames = sorted(
        [path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not frames:
        raise FileNotFoundError(f"No frames matched pattern {pattern!r} in {input_path}")
    return frames


def resolve_input_video(input_path: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES:
        return input_path

    if input_path.is_dir():
        videos = sorted(
            [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES],
            key=natural_key,
        )
        if len(videos) == 1:
            return videos[0]
        if not videos:
            raise FileNotFoundError(f"No video files found in {input_path}")
        raise ValueError(f"Expected one video file in {input_path}, found {len(videos)}")

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def extract_first_frame_from_video(video_path: Path, output_path: Path) -> Path:
    import av
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        print(f"Reusing existing source first frame: {output_path}")
        return output_path

    with av.open(str(video_path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                array = frame.to_ndarray(format="rgb24")
                Image.fromarray(np.asarray(array, dtype=np.uint8)).save(output_path)
                return output_path

    raise ValueError(f"Could not decode a frame from {video_path}")


def build_source_video(args: argparse.Namespace, input_path: Path, work_dir: Path) -> Path:
    output_path = work_dir / "source_video.mp4"
    if args.dataset == "gtacrime" or (input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES):
        source_video = resolve_input_video(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.is_file():
            shutil.copy2(source_video, output_path)
        else:
            print(f"Reusing existing source video copy: {output_path}")
        return output_path

    frame_pattern = resolve_frame_pattern(args.dataset, input_path, args.glob)
    collect_frames(input_path, frame_pattern)
    if output_path.is_file():
        print(f"Reusing existing source video: {output_path}")
        return output_path

    command = [
        sys.executable,
        str(SEQUENCE_TO_VIDEO_SCRIPT),
        "--dataset",
        args.dataset,
        "--input-dir",
        str(input_path),
        "--glob",
        frame_pattern,
        "--output",
        str(output_path),
        "--fps",
        str(args.reference_video_fps),
        "--crf",
        str(args.reference_video_crf),
    ]
    run_command(command)
    return output_path


def resolve_reference_image(args: argparse.Namespace, source_first_frame: Path, work_dir: Path) -> Path:
    if args.translated_first_frame:
        translated = Path(args.translated_first_frame).expanduser().resolve()
        if not translated.is_file():
            raise FileNotFoundError(f"translated-first-frame does not exist: {translated}")
        return translated

    if args.use_local_flux_runner:
        translated = work_dir / args.flux_output_name
        translated.parent.mkdir(parents=True, exist_ok=True)
        if translated.is_file():
            print(f"Reusing existing translated first frame: {translated}")
            return translated

        runner_path = Path(args.flux_runner_path).expanduser().resolve()
        if not runner_path.is_file():
            raise FileNotFoundError(f"FLUX runner not found: {runner_path}")

        command = [
            args.flux_python or sys.executable,
            str(runner_path),
            "--input",
            str(source_first_frame),
            "--output",
            str(translated),
            "--prompt",
            args.flux_prompt or args.prompt,
            "--negative-prompt",
            args.negative_prompt,
            "--seed",
            str(args.flux_seed),
            "--steps",
            str(args.flux_steps),
            "--guidance",
            str(args.flux_guidance),
            "--device",
            args.flux_device,
        ]
        if args.flux_model:
            command.extend(["--model", args.flux_model])
        if args.flux_no_cpu_offload:
            command.append("--no-cpu-offload")

        run_command(command)
        if not translated.is_file():
            raise FileNotFoundError(f"The local FLUX runner did not create {translated}")
        return translated

    if not args.flux_command_template:
        raise ValueError(
            "Provide --translated-first-frame, --use-local-flux-runner, or --flux-command-template "
            "so preprocessing can create a reference image."
        )

    translated = work_dir / args.flux_output_name
    translated.parent.mkdir(parents=True, exist_ok=True)
    if translated.is_file():
        print(f"Reusing existing translated first frame: {translated}")
        return translated

    command = args.flux_command_template.format(
        input_image=str(source_first_frame),
        output_image=str(translated),
        prompt=args.flux_prompt or args.prompt,
        negative_prompt=args.negative_prompt,
        work_dir=str(work_dir),
    )
    print("Executing FLUX first-frame command template.")
    subprocess.run(command, shell=True, check=True)
    if not translated.is_file():
        raise FileNotFoundError(f"The FLUX command did not create {translated}")
    return translated


def _import_control_helpers():
    from run_ic_lora_style_transfer import (
        create_depth_video_from_rgb,
        create_edge_video_from_rgb,
        is_readable_video_file,
        read_video_metadata,
    )
    from depth_control_adapter import create_gt_depth_control_video

    return (
        create_depth_video_from_rgb,
        create_edge_video_from_rgb,
        create_gt_depth_control_video,
        is_readable_video_file,
        read_video_metadata,
    )


def resolve_control_video(args: argparse.Namespace, source_video: Path, input_path: Path, work_dir: Path) -> Path:
    (
        create_depth_video_from_rgb,
        create_edge_video_from_rgb,
        create_gt_depth_control_video,
        is_readable_video_file,
        read_video_metadata,
    ) = _import_control_helpers()

    controls_dir = work_dir / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    if args.conditioning_video:
        control_video = Path(args.conditioning_video).expanduser().resolve()
        if not is_readable_video_file(control_video):
            raise FileNotFoundError(f"conditioning video does not exist or is unreadable: {control_video}")
        print(f"Using provided conditioning video: {control_video}")
        return control_video

    if args.conditioning_mode == "rgb":
        return source_video

    if args.accept_gt_depths:
        if args.conditioning_mode != "depth":
            raise ValueError("--accept-gt-depths requires --conditioning-mode depth")
        missing = [
            name
            for name, value in (
                ("--gt-depth-dir", args.gt_depth_dir),
                ("--gt-depth-source-root", args.gt_depth_source_root),
                ("--gt-depth-source-path", args.gt_depth_source_path),
            )
            if not value
        ]
        if missing:
            raise ValueError("--accept-gt-depths requires " + ", ".join(missing))

        control_video = Path(args.depth_output).expanduser().resolve() if args.depth_output else controls_dir / "gt_depth.mp4"
        if is_readable_video_file(control_video):
            print(f"Reusing existing GT-depth control video: {control_video}")
            return control_video

        _width, _height, frame_count, fps = read_video_metadata(str(source_video))
        create_gt_depth_control_video(
            dataset=args.gt_depth_dataset or args.dataset,
            gt_depth_root=args.gt_depth_dir,
            source_root=args.gt_depth_source_root,
            source_path=args.gt_depth_source_path or str(input_path),
            output_path=str(control_video),
            fps=fps,
            frame_cap=frame_count,
        )
        return control_video

    if args.conditioning_mode == "edge":
        control_video = Path(args.edge_output).expanduser().resolve() if args.edge_output else controls_dir / "edge.mp4"
        if is_readable_video_file(control_video):
            print(f"Reusing existing edge control video: {control_video}")
            return control_video
        create_edge_video_from_rgb(
            rgb_video_path=str(source_video),
            output_path=str(control_video),
            low_threshold=args.edge_low_threshold,
            high_threshold=args.edge_high_threshold,
        )
        return control_video

    if args.conditioning_mode != "depth":
        raise ValueError(f"Unsupported conditioning mode: {args.conditioning_mode}")

    if args.depth_backend != "video-depth-anything" and not args.depth_model:
        raise ValueError(
            "conditioning-mode=depth requires --depth-model unless --depth-backend video-depth-anything is used."
        )

    control_video = Path(args.depth_output).expanduser().resolve() if args.depth_output else controls_dir / "depth.mp4"
    if is_readable_video_file(control_video):
        print(f"Reusing existing depth control video: {control_video}")
        return control_video

    create_depth_video_from_rgb(
        rgb_video_path=str(source_video),
        output_path=str(control_video),
        depth_model_name_or_path=args.depth_model,
        cache_dir=args.depth_cache_dir,
        depth_backend=args.depth_backend,
        depth_device=args.depth_device,
        depth_use_fast=args.depth_use_fast,
        video_depth_anything_root=args.video_depth_anything_root,
        video_depth_anything_python=args.video_depth_anything_python,
        video_depth_anything_encoder=args.video_depth_anything_encoder,
        video_depth_anything_metric=args.video_depth_anything_metric,
        video_depth_anything_input_size=args.video_depth_anything_input_size,
        video_depth_anything_max_res=args.video_depth_anything_max_res,
        video_depth_anything_max_len=args.video_depth_anything_max_len,
        video_depth_anything_target_fps=args.video_depth_anything_target_fps,
        video_depth_anything_fp32=args.video_depth_anything_fp32,
    )
    return control_video


def prepare_style_transfer_inputs(args: argparse.Namespace) -> PreparedStyleTransferInputs:
    input_path = Path(args.input_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.accept_gt_depths:
        args.gt_depth_source_path = args.gt_depth_source_path or str(input_path)

    source_video = build_source_video(args, input_path, work_dir)
    source_first_frame = extract_first_frame_from_video(source_video, work_dir / "source_first_frame.png")
    reference_image = resolve_reference_image(args, source_first_frame, work_dir)
    control_video = resolve_control_video(args, source_video, input_path, work_dir)

    return PreparedStyleTransferInputs(
        source_video=str(source_video),
        source_first_frame=str(source_first_frame),
        reference_image=str(reference_image),
        control_video=str(control_video),
        work_dir=str(work_dir),
    )


def add_preprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="generic")
    parser.add_argument("--input-path", required=True, help="Source video file, video folder, or image-sequence folder.")
    parser.add_argument("--glob", default="auto", help="Frame glob for image-sequence inputs.")
    parser.add_argument("--work-dir", required=True, help="Directory for intermediate preprocessing artifacts.")
    parser.add_argument("--reference-video-fps", type=float, default=24.0)
    parser.add_argument("--reference-video-crf", type=int, default=18)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")

    parser.add_argument("--translated-first-frame", default=None)
    parser.add_argument("--use-local-flux-runner", action="store_true")
    parser.add_argument("--flux-runner-path", default=str(DEFAULT_FLUX_RUNNER))
    parser.add_argument("--flux-python", default=None)
    parser.add_argument("--flux-model", default=None)
    parser.add_argument("--flux-seed", type=int, default=42)
    parser.add_argument("--flux-steps", type=int, default=50)
    parser.add_argument("--flux-guidance", type=float, default=4.0)
    parser.add_argument("--flux-device", default="cuda:0")
    parser.add_argument("--flux-no-cpu-offload", action="store_true")
    parser.add_argument("--flux-prompt", default=None)
    parser.add_argument("--flux-command-template", default=None)
    parser.add_argument("--flux-output-name", default="flux/translated_first_frame.png")

    parser.add_argument("--conditioning-mode", choices=("rgb", "depth", "edge"), default="rgb")
    parser.add_argument("--conditioning-video", default=None)
    parser.add_argument("--depth-model", default=None)
    parser.add_argument("--depth-backend", choices=("auto", "image", "video-depth-anything"), default="auto")
    parser.add_argument("--depth-cache-dir", default=None)
    parser.add_argument("--depth-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--depth-use-fast", action="store_true")
    parser.add_argument("--depth-output", default=None)
    parser.add_argument("--edge-output", default=None)
    parser.add_argument("--edge-low-threshold", type=int, default=100)
    parser.add_argument("--edge-high-threshold", type=int, default=200)
    parser.add_argument("--accept-gt-depths", action="store_true")
    parser.add_argument("--gt-depth-dir", default=None)
    parser.add_argument("--gt-depth-source-root", default=None)
    parser.add_argument("--gt-depth-source-path", default=None)
    parser.add_argument("--gt-depth-dataset", default=None)
    parser.add_argument("--video-depth-anything-root", default=None)
    parser.add_argument("--video-depth-anything-python", default=None)
    parser.add_argument("--video-depth-anything-encoder", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--video-depth-anything-metric", action="store_true")
    parser.add_argument("--video-depth-anything-input-size", type=int, default=None)
    parser.add_argument("--video-depth-anything-max-res", type=int, default=None)
    parser.add_argument("--video-depth-anything-max-len", type=int, default=None)
    parser.add_argument("--video-depth-anything-target-fps", type=int, default=None)
    parser.add_argument("--video-depth-anything-fp32", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare source video, reference image, and control video for IC-LoRA.")
    add_preprocess_args(parser)
    parser.add_argument("--json-output", default=None, help="Optional path to write prepared artifact paths as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_style_transfer_inputs(args)
    payload = asdict(prepared)
    print(json.dumps(payload, indent=2))
    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
