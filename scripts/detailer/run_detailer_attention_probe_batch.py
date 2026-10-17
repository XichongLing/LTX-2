from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/data/detailer_experiments")
DEFAULT_OUTPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/Image_seq_transl/detailer_attention_probing")


@dataclass(frozen=True)
class StrengthSetting:
    name: str
    video_strength: float
    attention_strength: float
    image_strength: float


STRENGTH_SETTINGS = (
    StrengthSetting("s01_video1p0_attention1p0_image0p7", 1.0, 1.0, 0.7),
    StrengthSetting("s02_video1p0_attention1p0_image1p0", 1.0, 1.0, 1.0),
    StrengthSetting("s03_video0p3_attention1p0_image0p7", 0.3, 1.0, 0.7),
    StrengthSetting("s04_video1p0_attention0p3_image0p7", 1.0, 0.3, 0.7),
    StrengthSetting("s05_video0p3_attention0p3_image0p7", 0.3, 0.3, 0.7),
    StrengthSetting("s06_video1p0_attention0p0_image0p7", 1.0, 0.0, 0.7),
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTX-2 detailer attention-probing experiments.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--ref-video-dir", type=Path, default=None)
    parser.add_argument("--ref-img-dir", type=Path, default=None)
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
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument("--probe-layers", default="0,8,16,24,32,40,47")
    parser.add_argument("--probe-steps", default="all")
    parser.add_argument("--probe-heads", choices=("mean", "all"), default="mean")
    parser.add_argument("--probe-query-chunk-size", type=int, default=128)
    parser.add_argument(
        "--video-stem",
        action="append",
        default=None,
        help="Only run videos with this stem. Can be repeated.",
    )
    parser.add_argument("--quantization", default="fp8-cast", help="Quantization policy, or 'none'.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def find_video_files(ref_video_dir: Path) -> list[Path]:
    return sorted(path for path in ref_video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def find_matching_image(ref_img_dir: Path, stem: str) -> Path:
    matches = sorted(
        path for path in ref_img_dir.iterdir() if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not matches:
        raise FileNotFoundError(f"No reference image named {stem}.* found in {ref_img_dir}")
    return matches[0]


def build_command(args: argparse.Namespace, video_path: Path, image_path: Path, setting: StrengthSetting, run_dir: Path) -> list[str]:
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
        str(video_path),
        "--conditioning-video",
        str(video_path),
        "--output",
        str(run_dir / "output.mp4"),
        "--seed",
        str(args.seed),
        "--video-strength",
        str(setting.video_strength),
        "--conditioning-attention-strength",
        str(setting.attention_strength),
        "--image-condition",
        str(image_path),
        "0",
        str(setting.image_strength),
        str(args.image_crf),
        "--attention-probe-output",
        str(run_dir / "attention_probe.jsonl"),
        "--attention-probe-layers",
        args.probe_layers,
        "--attention-probe-steps",
        args.probe_steps,
        "--attention-probe-heads",
        args.probe_heads,
        "--attention-probe-query-chunk-size",
        str(args.probe_query_chunk_size),
        "--quiet-layer-streaming",
    ]
    if args.quantization.lower() != "none":
        command.extend(["--quantization", args.quantization])
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.negative_prompt:
        command.extend(["--negative-prompt", args.negative_prompt])
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


def write_metadata(run_dir: Path, video_path: Path, image_path: Path, setting: StrengthSetting, command: list[str]) -> None:
    metadata = {
        "video": str(video_path),
        "image": str(image_path),
        "setting": asdict(setting),
        "command": command,
        "created_at": time.time(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    ref_video_dir = (args.ref_video_dir or input_root / "ref_video").expanduser().resolve()
    ref_img_dir = (args.ref_img_dir or input_root / "ref_img").expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = find_video_files(ref_video_dir)
    if args.video_stem:
        wanted_stems = set(args.video_stem)
        videos = [video for video in videos if video.stem in wanted_stems]
    if not videos:
        raise FileNotFoundError(f"No video files found in {ref_video_dir}")

    failures: list[tuple[str, int]] = []
    total_runs = len(videos) * len(STRENGTH_SETTINGS)
    run_index = 0
    for video_path in videos:
        image_path = find_matching_image(ref_img_dir, video_path.stem)
        for setting in STRENGTH_SETTINGS:
            run_index += 1
            run_dir = output_root / video_path.stem / setting.name
            run_dir.mkdir(parents=True, exist_ok=True)
            output_mp4 = run_dir / "output.mp4"
            probe_jsonl = run_dir / "attention_probe.jsonl"
            if args.skip_existing and output_mp4.exists() and probe_jsonl.exists():
                print(f"[{run_index}/{total_runs}] Skipping existing {run_dir}")
                continue

            command = build_command(args, video_path, image_path, setting, run_dir)
            write_metadata(run_dir, video_path, image_path, setting, command)
            print(f"[{run_index}/{total_runs}] Running {video_path.stem} {setting.name}")
            print(f"  output: {output_mp4}")
            print(f"  probe:  {probe_jsonl}")
            if args.dry_run:
                print("  dry-run command:")
                print(" ".join(command))
                continue

            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
                result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)

            if result.returncode != 0:
                failures.append((str(run_dir), result.returncode))
                print(f"  failed with exit code {result.returncode}; see {run_dir / 'run.log'}")
                if args.stop_on_error:
                    break
            else:
                print("  done")
        if failures and args.stop_on_error:
            break

    summary = {
        "total_runs": total_runs,
        "failures": [{"run_dir": run_dir, "returncode": returncode} for run_dir, returncode in failures],
        "completed_at": time.time(),
    }
    (output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
