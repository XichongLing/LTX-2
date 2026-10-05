#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for package in (REPO_ROOT / "packages/ltx-core/src", REPO_ROOT / "packages/ltx-pipelines/src"):
    sys.path.insert(0, str(package))

from ltx_pipelines.correspondence_mask import load_file_mask

DEFAULT_PROMPT = (
    "Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in "
    "the appearance, color, materials, lighting, and style of the reference image. Preserve the original "
    "composition, shot timing, scene geometry, and character identity while using the RGB source video for structure."
)
DEFAULT_NEGATIVE_PROMPT = (
    "flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, "
    "compression artifacts"
)
DEFAULT_SEG_ROOT = Path("/media/xichong-drz-inokko/4715db97-9866-4ec1-9eab-1f836151d77b1/segmentation")
DEFAULT_SAM3_PYTHON = DEFAULT_SEG_ROOT / "envs/sam3/bin/python"
DEFAULT_SAM3_TOOL = DEFAULT_SEG_ROOT / "tools/segment_sam3_text_video.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3-mask-driven multiscale Stage-1 and region-selective Stage-2.")
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--stage1-mask-concept", required=True)
    parser.add_argument("--stage2-mask-concept", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--sam3-output-root", default=None)
    parser.add_argument("--stage2-mode", choices=("video", "dual", "spatial"), default="dual")
    parser.add_argument("--stage2-kv-mode", choices=("none", "partition", "append"), default="partition")
    parser.add_argument("--stage2-g", type=float, default=0.0)
    parser.add_argument("--rerun-sam3", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/manifests but do not execute GPU/SAM3 commands.")

    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--ltx-python", default=None)
    parser.add_argument("--sam3-python", default=str(DEFAULT_SAM3_PYTHON))
    parser.add_argument("--sam3-tool", default=str(DEFAULT_SAM3_TOOL))
    parser.add_argument("--distilled-checkpoint-path", default=str(REPO_ROOT / "models/ltx-2.3-22b-distilled-1.1.safetensors"))
    parser.add_argument("--spatial-upsampler-path", default=str(REPO_ROOT / "models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
    parser.add_argument("--gemma-root", default="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized")
    parser.add_argument("--ic-lora-path", default=str(REPO_ROOT / "models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"))
    parser.add_argument("--quantization", default="fp8-cast")
    parser.add_argument("--inside-scale", type=int, default=4)
    parser.add_argument("--outside-scale", type=int, default=2)
    parser.add_argument("--partition-threshold", type=float, default=0.5)
    parser.add_argument("--inside-strength", type=float, default=1.0)
    parser.add_argument("--outside-strength", type=float, default=1.0)
    parser.add_argument("--stage-1-ic-lora-strength", type=float, default=1.0)
    parser.add_argument("--stage-2-ic-lora-strength", type=float, default=1.0)
    parser.add_argument("--stage-2-noise-seed", type=int, default=None)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--stage-2-conditioning-attention-strength", type=float, default=1.0)
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "mask"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_video(path: str | Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(raw)["streams"][0]
    fps_expr = stream.get("avg_frame_rate") or "0/1"
    if "/" in fps_expr:
        n, d = fps_expr.split("/", 1)
        fps = float(n) / float(d) if float(d) else 0.0
    else:
        fps = float(fps_expr)
    frame_count = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else int(round(float(stream.get("duration") or 0.0) * fps))
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps, "frame_count": frame_count}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], *, log_path: Path, dry_run: bool = False, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = command_to_string(command) + "\n"
    log_path.write_text(command_text)
    if log_path.name == "run.log":
        (log_path.parent / "command.txt").write_text(command_text)
    elif log_path.name.endswith(".log"):
        (log_path.parent / f"{log_path.stem}_command.txt").write_text(command_text)
    if dry_run:
        return 0
    with log_path.open("a") as log:
        proc = subprocess.Popen(command, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def expected_mask_dir(sam3_root: Path, prefix: str, concept: str) -> Path:
    return sam3_root / f"{prefix}_{slugify(concept)}" / "masks"


def mask_count(mask_dir: Path) -> int:
    return len(sorted(mask_dir.glob("*.png")))


def ensure_sam3_masks(
    *,
    video: Path,
    concept: str,
    output_dir: Path,
    args: argparse.Namespace,
    expected_frames: int,
    commands: dict[str, str],
    key: str,
) -> Path:
    mask_dir = output_dir / "masks"
    if mask_dir.is_dir() and mask_count(mask_dir) >= expected_frames and not args.rerun_sam3:
        commands[key] = "reused existing masks"
        return mask_dir
    command = [
        str(Path(args.sam3_python).expanduser().resolve()),
        str(Path(args.sam3_tool).expanduser().resolve()),
        "--video",
        str(video),
        "--text-prompt",
        concept,
        "--output-dir",
        str(output_dir),
        "--all-instances",
        "--fps",
        "source",
        "--overlay-alpha",
        "0.45",
        "--overwrite",
    ]
    commands[key] = command_to_string(command)
    rc = run_command(command, log_path=output_dir / "sam3.log", dry_run=args.dry_run)
    if rc != 0:
        raise RuntimeError(f"SAM3 command failed for {key} with exit code {rc}")
    if not args.dry_run and mask_count(mask_dir) < expected_frames:
        raise RuntimeError(f"SAM3 mask dir {mask_dir} has {mask_count(mask_dir)} frames, expected {expected_frames}")
    return mask_dir


def ltx_common_args(args: argparse.Namespace, repo: Path) -> list[str]:
    return [
        str(Path(args.ltx_python).expanduser().resolve() if args.ltx_python else repo / ".venv/bin/python3"),
        str(repo / "scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"),
        "--distilled-checkpoint-path",
        str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        "--spatial-upsampler-path",
        str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        "--gemma-root",
        str(Path(args.gemma_root).expanduser().resolve()),
        "--ic-lora-path",
        str(Path(args.ic_lora_path).expanduser().resolve()),
        "--reference-video",
        str(Path(args.source_video).expanduser().resolve()),
        "--conditioning-video",
        str(Path(args.source_video).expanduser().resolve()),
        "--reference-image",
        str(Path(args.reference_image).expanduser().resolve()),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--seed",
        str(args.seed),
        "--stage-2-noise-seed",
        str(args.stage_2_noise_seed if args.stage_2_noise_seed is not None else args.seed),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num-frames",
        str(args.num_frames),
        "--frame-rate",
        str(args.frame_rate),
        "--image-strength",
        str(args.image_strength),
        "--image-crf",
        str(args.image_crf),
        "--video-strength",
        str(args.video_strength),
        "--conditioning-mode",
        "rgb",
        "--conditioning-attention-strength",
        str(args.conditioning_attention_strength),
        "--stage-2-conditioning-attention-strength",
        str(args.stage_2_conditioning_attention_strength),
        "--stage-1-ic-lora-strength",
        str(args.stage_1_ic_lora_strength),
        "--stage-2-ic-lora-strength",
        str(args.stage_2_ic_lora_strength),
        "--first-frame-attention-schedule",
        "constant",
        "--first-frame-attention-multiplier",
        "1.0",
        "--source-strength-schedule",
        "constant",
        "--source-strength",
        "1.0",
        "--source-strength-routing",
        "target-queries-only",
        "--reference-downscale-factor-override",
        "2",
        "--quantization",
        args.quantization,
    ]


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or (args.num_frames - 1) % 8:
        raise ValueError("num_frames must follow 8k+1 for the current LTX causal layout")
    if args.stage2_mode in {"dual", "spatial"} and not 0.0 <= args.stage2_g <= 1.0:
        raise ValueError("--stage2-g must be in [0,1]")
    if args.stage2_mode != "dual" and args.stage2_kv_mode != "none":
        raise ValueError("--stage2-kv-mode is only meaningful with --stage2-mode dual")

    repo = Path(args.repo_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    source_video = Path(args.source_video).expanduser().resolve()
    reference_image = Path(args.reference_image).expanduser().resolve()
    sam3_root = Path(args.sam3_output_root).expanduser().resolve() if args.sam3_output_root else output_root / "sam3"
    stage2_noise_seed = args.stage_2_noise_seed if args.stage_2_noise_seed is not None else args.seed

    source_meta = ffprobe_video(source_video)
    if source_meta["frame_count"] < args.num_frames:
        raise ValueError(f"source has {source_meta['frame_count']} frames, requested {args.num_frames}")

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_link_or_copy(source_video, output_root / "inputs" / source_video.name)
    ensure_link_or_copy(reference_image, output_root / "inputs" / reference_image.name)

    commands: dict[str, str] = {}
    stage1_mask_dir = ensure_sam3_masks(
        video=source_video,
        concept=args.stage1_mask_concept,
        output_dir=sam3_root / f"stage1_{slugify(args.stage1_mask_concept)}",
        args=args,
        expected_frames=args.num_frames,
        commands=commands,
        key="sam3_stage1",
    )

    # Validate that the mask loader accepts the SAM3 folder under the exact inference resize/crop convention.
    if not args.dry_run:
        _ = load_file_mask(path=stage1_mask_dir, num_frames=args.num_frames, height=args.height // 2, width=args.width // 2)

    partition_dir = output_root / "partition"
    partition_cmd = [
        str(Path(args.ltx_python).expanduser().resolve() if args.ltx_python else repo / ".venv/bin/python3"),
        str(repo / "scripts/exp/two_scale/build_reference_partition.py"),
        "--mask",
        str(stage1_mask_dir),
        "--output-dir",
        str(partition_dir),
        "--num-frames",
        str(args.num_frames),
        "--stage1-width",
        str(args.width // 2),
        "--stage1-height",
        str(args.height // 2),
        "--fps",
        str(args.frame_rate),
        "--inside-scale",
        str(args.inside_scale),
        "--outside-scale",
        str(args.outside_scale),
        "--threshold",
        str(args.partition_threshold),
        "--inside-strength",
        str(args.inside_strength),
        "--outside-strength",
        str(args.outside_strength),
    ]
    commands["build_partition"] = command_to_string(partition_cmd)
    rc = run_command(partition_cmd, log_path=partition_dir / "build_partition.log", dry_run=args.dry_run, cwd=repo)
    if rc != 0:
        raise RuntimeError(f"partition builder failed with exit code {rc}")
    partition_config = partition_dir / "ref_partition_config.json"

    common = ltx_common_args(args, repo)
    stage1_snapshot = output_root / "stage1_snapshot" / "output.mp4"
    stage1_cmd = common + [
        "--ref-partition-config",
        str(partition_config),
        "--stage-2-ic-lora-strength",
        "0.0",
        "--skip-stage-2",
        "--output",
        str(stage1_snapshot),
    ]
    commands["stage1_snapshot"] = command_to_string(stage1_cmd)
    rc = run_command(stage1_cmd, log_path=stage1_snapshot.parent / "run.log", dry_run=args.dry_run, cwd=repo)
    if rc != 0:
        raise RuntimeError(f"Stage-1 snapshot command failed with exit code {rc}")

    stage2_mask_dir = ensure_sam3_masks(
        video=stage1_snapshot,
        concept=args.stage2_mask_concept,
        output_dir=sam3_root / f"stage2_{slugify(args.stage2_mask_concept)}",
        args=args,
        expected_frames=args.num_frames,
        commands=commands,
        key="sam3_stage2",
    )
    if not args.dry_run:
        _ = load_file_mask(path=stage2_mask_dir, num_frames=args.num_frames, height=args.height, width=args.width)

    cache = output_root / "stage2_cache" / "stage2_input_cache.safetensors"
    baseline_dir = output_root / "runs" / "video_baseline"
    baseline_cmd = common + [
        "--source-strength-log",
        str(baseline_dir / "source_strength_log.json"),
        "--ref-partition-config",
        str(partition_config),
        "--save-stage-2-input",
        str(cache),
        "--stage-2-branch-mode",
        "video",
        "--stage-2-lora-execution",
        "runtime",
        "--output",
        str(baseline_dir / "output.mp4"),
    ]
    commands["stage2_video_baseline"] = command_to_string(baseline_cmd)
    rc = run_command(baseline_cmd, log_path=baseline_dir / "run.log", dry_run=args.dry_run, cwd=repo)
    if rc != 0:
        raise RuntimeError(f"Stage-2 video baseline command failed with exit code {rc}")

    selective_dir = output_root / "runs" / ("dual_partition" if args.stage2_mode == "dual" else f"{args.stage2_mode}_routing")
    selective_cmd = common + [
        "--source-strength-log",
        str(selective_dir / "source_strength_log.json"),
        "--ref-partition-config",
        str(partition_config),
        "--load-stage-2-input",
        str(cache),
        "--load-stage-2-input-ignore-metadata",
        "--stage-2-branch-mode",
        args.stage2_mode,
        "--stage-2-lora-execution",
        "runtime",
        "--output",
        str(selective_dir / "output.mp4"),
    ]
    if args.stage2_mode in {"dual", "spatial"}:
        selective_cmd += [
            "--stage-2-routing-mask",
            str(stage2_mask_dir),
            "--stage-2-dress-video-contribution",
            str(args.stage2_g),
        ]
    if args.stage2_mode == "dual":
        selective_cmd += [
            "--stage-2-dual-image-ic-lora",
            "--stage-2-kv-mode",
            args.stage2_kv_mode,
            "--stage-2-kv-strength",
            "1.0",
            "--stage-2-kv-layers",
            "all",
            "--stage-2-kv-cache-backend",
            "cpu",
        ]
    commands["stage2_region_selective"] = command_to_string(selective_cmd)
    rc = run_command(selective_cmd, log_path=selective_dir / "run.log", dry_run=args.dry_run, cwd=repo)
    if rc != 0:
        raise RuntimeError(f"Stage-2 region-selective command failed with exit code {rc}")

    outputs = {
        "stage1_snapshot": str(stage1_snapshot),
        "stage2_cache": str(cache),
        "video_baseline": str(baseline_dir / "output.mp4"),
        "region_selective": str(selective_dir / "output.mp4"),
    }
    validations: dict[str, Any] = {}
    if not args.dry_run:
        for label, video in {
            "stage1_snapshot": stage1_snapshot,
            "video_baseline": baseline_dir / "output.mp4",
            "region_selective": selective_dir / "output.mp4",
        }.items():
            meta = ffprobe_video(video)
            validations[label] = meta
            expected_w = args.width // 2 if label == "stage1_snapshot" else args.width
            expected_h = args.height // 2 if label == "stage1_snapshot" else args.height
            if meta["width"] != expected_w or meta["height"] != expected_h or meta["frame_count"] != args.num_frames:
                raise RuntimeError(f"unexpected {label} metadata: {meta}, expected {expected_w}x{expected_h} {args.num_frames} frames")
        baseline_manifest = json.loads((baseline_dir / "output.mp4.stage2.json").read_text())
        selective_manifest = json.loads((selective_dir / "output.mp4.stage2.json").read_text())
        if baseline_manifest.get("stage_2_branch_mode") != "video":
            raise RuntimeError("video baseline did not use controlled stage_2_branch_mode=video")
        if baseline_manifest.get("stage_2_input_video_checksum") != selective_manifest.get("stage_2_input_video_checksum"):
            raise RuntimeError("baseline and region-selective run used different Stage-2 video input checksums")
        outputs.update({
            "video_baseline_sha256": sha256_file(baseline_dir / "output.mp4"),
            "region_selective_sha256": sha256_file(selective_dir / "output.mp4"),
            "stage2_cache_sha256": sha256_file(cache),
        })

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "source_video": str(source_video),
        "source_video_sha256": None if args.dry_run else sha256_file(source_video),
        "reference_image": str(reference_image),
        "reference_image_sha256": None if args.dry_run else sha256_file(reference_image),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "stage_2_noise_seed": stage2_noise_seed,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "stage1_mask_concept": args.stage1_mask_concept,
        "stage2_mask_concept": args.stage2_mask_concept,
        "stage1_mask_dir": str(stage1_mask_dir),
        "stage2_mask_dir": str(stage2_mask_dir),
        "partition_config": str(partition_config),
        "stage2_mode": args.stage2_mode,
        "stage2_kv_mode": args.stage2_kv_mode,
        "stage2_g": args.stage2_g,
        "commands": commands,
        "outputs": outputs,
        "validations": validations,
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
