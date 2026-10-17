#!/usr/bin/env python3
"""Generate IC-LoRA reference videos from Intrinsic high-resolution albedo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

INTRINSIC_ROOT = Path("/home/xichong-drz-inokko/experimental/Intrinsic")
if str(INTRINSIC_ROOT) not in sys.path:
    sys.path.insert(0, str(INTRINSIC_ROOT))

from intrinsic.pipeline import load_models, run_pipeline  # noqa: E402


TORCH_HUB_TRUSTED_REPOS = ("rwightman_gen-efficientnet-pytorch",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create albedo reference videos for IC-LoRA training using experimental/Intrinsic."
    )
    parser.add_argument("dataset_json", type=Path, help="Dataset JSON with caption/media_path rows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Reference video directory. Defaults to <dataset-dir>/references_albedo.",
    )
    parser.add_argument("--weights", choices=("v2", "v2.1"), default="v2.1")
    parser.add_argument("--base-size", type=int, default=384)
    parser.add_argument(
        "--resize-conf",
        type=float,
        default=0.0,
        help="Intrinsic resize confidence. Use -1 for rounded original-size model input.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--override", action="store_true", help="Regenerate existing reference videos.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep per-frame PNG folders next to the output videos.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_json = args.dataset_json.resolve()
    dataset_dir = dataset_json.parent
    output_dir = (args.output_dir or dataset_dir / "references_albedo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resize_conf = None if args.resize_conf < 0 else args.resize_conf

    with dataset_json.open("r", encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    pretrust_torch_hub_repos()
    print(f"Loading Intrinsic {args.weights} models on {args.device} through albedo stage...", flush=True)
    models = load_models(args.weights, stage="albedo", device=args.device)

    updated: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        media_path = dataset_dir / sample["media_path"]
        reference_path = output_dir / f"{media_path.stem}_albedo.mp4"
        rel_reference_path = reference_path.relative_to(dataset_dir)
        print(f"[{index}/{len(samples)}] {media_path.name} -> {rel_reference_path}", flush=True)

        if args.override or not reference_path.exists():
            generate_albedo_video(
                media_path=media_path,
                reference_path=reference_path,
                models=models,
                resize_conf=resize_conf,
                base_size=args.base_size,
                device=args.device,
                crf=args.crf,
                max_frames=args.max_frames,
                keep_frames=args.keep_frames,
            )

        row = dict(sample)
        row["reference_path"] = str(rel_reference_path)
        updated.append(row)

    if args.max_samples is None and args.max_frames is None:
        with dataset_json.open("w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Updated {dataset_json} with albedo reference_path for {len(updated)} samples.")
    else:
        print("Smoke-test mode used; dataset JSON was not modified.")
    return 0


def pretrust_torch_hub_repos() -> None:
    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)
    trusted_list = hub_dir / "trusted_list"
    existing = set()
    if trusted_list.exists():
        existing = {line.strip() for line in trusted_list.read_text().splitlines()}
    with trusted_list.open("a") as file:
        for repo in TORCH_HUB_TRUSTED_REPOS:
            if repo not in existing:
                file.write(repo + "\n")


def generate_albedo_video(
    media_path: Path,
    reference_path: Path,
    models: dict[str, Any],
    resize_conf: float | None,
    base_size: int,
    device: str,
    crf: int,
    max_frames: int | None,
    keep_frames: bool,
) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = reference_path.with_name(reference_path.stem + ".tmp" + reference_path.suffix)
    frames_dir = reference_path.with_suffix("")
    temp_frames_dir = frames_dir.with_name(frames_dir.name + "_tmp_frames")

    if temp_path.exists():
        temp_path.unlink()
    if temp_frames_dir.exists():
        shutil.rmtree(temp_frames_dir)
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {media_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames) if frame_count else max_frames

    produced_frames = 0
    try:
        with torch.inference_mode(), tqdm(total=frame_count if frame_count > 0 else None) as pbar:
            while frame_count <= 0 or produced_frames < frame_count:
                ok, bgr = cap.read()
                if not ok:
                    break
                height, width = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                results = run_pipeline(
                    models,
                    rgb,
                    stage=3,
                    resize_conf=resize_conf,
                    base_size=base_size,
                    device=device,
                )
                albedo = np.clip(results["hr_alb"], 0.0, 1.0)
                if albedo.shape[:2] != (height, width):
                    albedo = cv2.resize(albedo, (width, height), interpolation=cv2.INTER_LINEAR)
                albedo_srgb = np.clip(albedo ** (1.0 / 2.2), 0.0, 1.0)
                out_bgr = cv2.cvtColor((albedo_srgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(temp_frames_dir / f"{produced_frames:06d}.png"), out_bgr)
                produced_frames += 1
                pbar.update(1)
    finally:
        cap.release()

    if produced_frames == 0:
        raise RuntimeError(f"No albedo frames produced for {media_path}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.6f}",
            "-start_number",
            "0",
            "-i",
            str(temp_frames_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            str(temp_path),
        ],
        check=True,
    )

    os.replace(temp_path, reference_path)
    if keep_frames:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        os.replace(temp_frames_dir, frames_dir)
    else:
        shutil.rmtree(temp_frames_dir)


if __name__ == "__main__":
    raise SystemExit(main())
