from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/data/detailer_experiments")
DEFAULT_OUTPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/Image_seq_transl/replace_non_first_frame")


@dataclass(frozen=True)
class VideoCase:
    name: str
    input_video: Path
    reference_image: Path


@dataclass(frozen=True)
class AnchorPosition:
    label: str
    frame_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detailer IC-LoRA single-image in-place replacement at first/intermediate/last frame positions."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--python",
        default=os.environ.get("LTX_PYTHON", str(REPO_ROOT / ".venv" / "bin" / "python3")),
        help="Python executable used to invoke run_ic_lora_detailer.py.",
    )
    parser.add_argument(
        "--distilled-checkpoint-path",
        default="/home/xichong-drz-inokko/Downloads/ltx-2-19b-distilled.safetensors",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        default="/home/xichong-drz-inokko/experimental/ComfyUI/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    )
    parser.add_argument(
        "--gemma-root",
        default="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--detailer-lora-path",
        default="/home/xichong-drz-inokko/experimental/ComfyUI/models/loras/LTX-2/ltx-2-19b-ic-lora-detailer.safetensors",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--detailer-lora-strength", type=float, default=1.0)
    parser.add_argument("--quantization", default="fp8-cast", help="Quantization policy, or 'none'.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument(
        "--matched-reference-pattern",
        default=None,
        help=(
            "Optional filename pattern in ref_img for per-frame reference images. "
            "Available fields: {case}, {frame_idx}, {frame_idx04}. "
            "Example: {case}_frame_{frame_idx04}_flux2.png"
        ),
    )
    return parser.parse_args()


def read_video_frame_count(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream found in {video_path}")
        if stream.frames and stream.frames > 0:
            return int(stream.frames)
        return sum(1 for packet in container.demux(stream) for _frame in packet.decode())


def discover_cases(input_root: Path) -> list[VideoCase]:
    ref_video_dir = input_root / "ref_video"
    ref_img_dir = input_root / "ref_img"
    image_by_stem = {path.stem: path for path in sorted(ref_img_dir.iterdir()) if path.is_file()}
    cases = []
    for video_path in sorted(ref_video_dir.iterdir()):
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            continue
        reference_image = image_by_stem.get(video_path.stem)
        if reference_image is None:
            raise FileNotFoundError(f"No reference image matching video stem {video_path.stem!r}")
        cases.append(VideoCase(video_path.stem, video_path, reference_image))
    if not cases:
        raise FileNotFoundError(f"No MP4 reference videos found in {ref_video_dir}")
    return cases


def select_positions(num_frames: int) -> list[AnchorPosition]:
    if num_frames < 4:
        raise ValueError(f"Need at least 4 frames to select first, two intermediate, and last positions: {num_frames}")
    return [
        AnchorPosition("first", 0),
        AnchorPosition("intermediate_a", max(1, round((num_frames - 1) / 3))),
        AnchorPosition("intermediate_b", min(num_frames - 2, round(2 * (num_frames - 1) / 3))),
        AnchorPosition("last", num_frames - 1),
    ]


def resolve_reference_image(args: argparse.Namespace, case: VideoCase, frame_idx: int) -> Path:
    if args.matched_reference_pattern is None:
        return case.reference_image

    filename = args.matched_reference_pattern.format(
        case=case.name,
        frame_idx=frame_idx,
        frame_idx04=f"{frame_idx:04d}",
    )
    reference_image = case.reference_image.parent / filename
    if not reference_image.is_file():
        raise FileNotFoundError(f"Matched reference image not found: {reference_image}")
    return reference_image


def build_command(
    args: argparse.Namespace,
    case: VideoCase,
    run_dir: Path,
    frame_idx: int,
    reference_image: Path,
) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "detailer" / "run_ic_lora_detailer.py"),
        "--distilled-checkpoint-path",
        str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        "--spatial-upsampler-path",
        str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        "--gemma-root",
        str(Path(args.gemma_root).expanduser().resolve()),
        "--detailer-lora-path",
        str(Path(args.detailer_lora_path).expanduser().resolve()),
        "--input-video",
        str(case.input_video.expanduser().resolve()),
        "--conditioning-video",
        str(case.input_video.expanduser().resolve()),
        "--output",
        str(run_dir / "output.mp4"),
        "--seed",
        str(args.seed),
        "--video-strength",
        str(args.video_strength),
        "--conditioning-attention-strength",
        str(args.conditioning_attention_strength),
        "--detailer-lora-strength",
        str(args.detailer_lora_strength),
        "--reference-image-replace",
        str(frame_idx),
        "--image-condition",
        str(reference_image.expanduser().resolve()),
        str(frame_idx),
        str(args.image_strength),
        str(args.image_crf),
        "--quiet-layer-streaming",
    ]
    if args.quantization.lower() != "none":
        command.extend(["--quantization", args.quantization])
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
    return command


def write_metadata(
    run_dir: Path,
    case: VideoCase,
    position: AnchorPosition,
    reference_image: Path,
    video_frame_count: int,
    command: list[str],
) -> None:
    metadata = {
        "case": asdict(case),
        "position": asdict(position),
        "reference_image": str(reference_image),
        "video_frame_count": video_frame_count,
        "command": command,
        "created_at": time.time(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected_cases = set(args.case or [])
    failures: list[tuple[Path, int]] = []

    cases = [case for case in discover_cases(input_root) if not selected_cases or case.name in selected_cases]
    for case in cases:
        video_frame_count = args.num_frames or read_video_frame_count(case.input_video.expanduser().resolve())
        for position in select_positions(video_frame_count):
            run_name = f"{position.label}_frame_{position.frame_idx:04d}"
            run_dir = output_root / case.name / run_name
            reference_image = resolve_reference_image(args, case, position.frame_idx)
            command = build_command(args, case, run_dir, position.frame_idx, reference_image)
            run_dir.mkdir(parents=True, exist_ok=True)
            write_metadata(run_dir, case, position, reference_image, video_frame_count, command)

            if args.skip_existing and (run_dir / "output.mp4").exists():
                print(f"Skipping existing {run_dir / 'output.mp4'}")
                continue
            if args.dry_run:
                print(" ".join(command))
                continue

            print(f"Running {case.name} {position.label} frame={position.frame_idx}")
            with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
                result = subprocess.run(command, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                failures.append((run_dir, result.returncode))
                print(f"FAILED {run_dir} returncode={result.returncode}", file=sys.stderr)
                if args.stop_on_error:
                    break
        if failures and args.stop_on_error:
            break

    if failures:
        for run_dir, returncode in failures:
            print(f"Failure: {run_dir} returncode={returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
