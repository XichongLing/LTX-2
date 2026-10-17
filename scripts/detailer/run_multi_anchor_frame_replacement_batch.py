from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/Image_seq_transl/multi_anchor_frames")
DEFAULT_FRAME_COUNTS = (1, 3, 5, 10, 20)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEMPORAL_SCALE = 8


@dataclass(frozen=True)
class VideoCase:
    name: str
    input_video: Path
    reference_frame_dir: Path


VIDEO_CASES = (
    VideoCase(
        "anna",
        Path("/home/xichong-drz-inokko/UniRelight/outputs/albedo_video_examples/anna_light-0000_720.mp4"),
        Path("/home/xichong-drz-inokko/UniRelight/inputs/albedo_video_frames/anna"),
    ),
    VideoCase(
        "arrival",
        Path("/home/xichong-drz-inokko/UniRelight/outputs/albedo_video_examples/arrival2_light-0000_720.mp4"),
        Path("/home/xichong-drz-inokko/UniRelight/inputs/albedo_video_frames/arrival2"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare detailer IC-LoRA in-place replacement across multiple reference-frame anchor counts."
    )
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
    parser.add_argument("--frame-count", type=int, action="append", default=None)
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
    parser.add_argument("--case", choices=[case.name for case in VIDEO_CASES], action="append", default=None)
    parser.add_argument(
        "--run-duplicate-caps",
        action="store_true",
        help="Run requested counts that cap to an already-run actual anchor set.",
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


def frame_index(path: Path) -> int:
    match = re.match(r"(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame index from {path.name}")
    return int(match.group(1))


def list_reference_frames(frame_dir: Path) -> dict[int, Path]:
    frames = {}
    for path in sorted(frame_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frames[frame_index(path)] = path
    if not frames:
        raise FileNotFoundError(f"No reference frames found in {frame_dir}")
    return frames


def pixel_frame_to_latent_idx(frame_idx: int, num_frames: int) -> int:
    if frame_idx == 0:
        return 0
    latent_idx = ((frame_idx - 1) // TEMPORAL_SCALE) + 1
    latent_count = ((num_frames - 1) // TEMPORAL_SCALE) + 1
    return min(latent_idx, latent_count - 1)


def select_anchor_indices(requested_count: int, available_indices: list[int], num_frames: int) -> tuple[list[int], bool]:
    candidates = [idx for idx in available_indices if idx < num_frames]
    by_latent: dict[int, int] = {}
    for idx in candidates:
        latent_idx = pixel_frame_to_latent_idx(idx, num_frames)
        by_latent.setdefault(latent_idx, idx)

    latent_indices = sorted(by_latent)
    actual_count = min(requested_count, len(latent_indices))
    if actual_count <= 0:
        raise ValueError("No anchor frames overlap the requested video frame range.")
    if actual_count == 1:
        chosen_latents = [latent_indices[0]]
    else:
        max_pos = len(latent_indices) - 1
        chosen_positions = sorted({round(i * max_pos / (actual_count - 1)) for i in range(actual_count)})
        chosen_latents = [latent_indices[pos] for pos in chosen_positions]
    return [by_latent[latent_idx] for latent_idx in chosen_latents], actual_count < requested_count


def build_command(
    args: argparse.Namespace,
    case: VideoCase,
    run_dir: Path,
    anchor_indices: list[int],
    frames_by_index: dict[int, Path],
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
        *[str(idx) for idx in anchor_indices],
        "--quiet-layer-streaming",
    ]
    for idx in anchor_indices:
        command.extend(
            [
                "--image-condition",
                str(frames_by_index[idx].expanduser().resolve()),
                str(idx),
                str(args.image_strength),
                str(args.image_crf),
            ]
        )
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
    requested_count: int,
    anchor_indices: list[int],
    capped: bool,
    num_frames: int,
    command: list[str],
) -> None:
    metadata = {
        "case": asdict(case),
        "requested_anchor_count": requested_count,
        "actual_anchor_count": len(anchor_indices),
        "capped_by_distinct_latent_frames": capped,
        "temporal_scale": TEMPORAL_SCALE,
        "num_frames": num_frames,
        "anchor_indices": anchor_indices,
        "anchor_latent_indices": [pixel_frame_to_latent_idx(idx, num_frames) for idx in anchor_indices],
        "command": command,
        "created_at": time.time(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    frame_counts = args.frame_count or list(DEFAULT_FRAME_COUNTS)
    selected_case_names = set(args.case or [case.name for case in VIDEO_CASES])
    cases = [case for case in VIDEO_CASES if case.name in selected_case_names]
    failures: list[tuple[Path, int]] = []

    for case in cases:
        frames_by_index = list_reference_frames(case.reference_frame_dir.expanduser().resolve())
        video_frame_count = args.num_frames or read_video_frame_count(case.input_video.expanduser().resolve())
        seen_anchor_sets: set[tuple[int, ...]] = set()
        for requested_count in frame_counts:
            anchor_indices, capped = select_anchor_indices(requested_count, sorted(frames_by_index), video_frame_count)
            anchor_key = tuple(anchor_indices)
            run_name = f"anchors_requested_{requested_count:02d}_actual_{len(anchor_indices):02d}"
            run_dir = output_root / case.name / run_name
            command = build_command(args, case, run_dir, anchor_indices, frames_by_index)
            run_dir.mkdir(parents=True, exist_ok=True)
            write_metadata(run_dir, case, requested_count, anchor_indices, capped, video_frame_count, command)

            if anchor_key in seen_anchor_sets and not args.run_duplicate_caps:
                print(f"Skipping duplicate capped anchor set for {case.name} requested={requested_count}: {anchor_indices}")
                continue
            seen_anchor_sets.add(anchor_key)

            if args.skip_existing and (run_dir / "output.mp4").exists():
                print(f"Skipping existing {run_dir / 'output.mp4'}")
                continue
            if args.dry_run:
                print(" ".join(command))
                continue

            print(f"Running {case.name} requested={requested_count} actual={len(anchor_indices)} anchors={anchor_indices}")
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
