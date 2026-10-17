#!/usr/bin/env python3
"""Tracked-region DINOv2 margins for the Anna dress-collapse study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CHECKPOINTS = (0, 24, 48, 72, 96, 120)
REGIONS = {
    # Coordinates were drawn on the 1920x804 edited/source first frame.
    "upper": np.asarray(((0.37, 0.30), (0.48, 0.28), (0.52, 0.58), (0.45, 0.68), (0.37, 0.60)), np.float32),
    "lower": np.asarray(((0.08, 0.53), (0.34, 0.40), (0.47, 0.65), (0.34, 0.88), (0.08, 0.80)), np.float32),
}
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) <= CHECKPOINTS[-1]:
        raise ValueError(f"{path} has only {len(frames)} frames")
    return frames


def track_regions(source: list[np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    height, width = source[0].shape[:2]
    work_size = (640, max(1, round(640 * height / width)))
    points = {name: polygon * np.asarray((work_size[0], work_size[1])) for name, polygon in REGIONS.items()}
    tracked = {name: {0: value.copy()} for name, value in points.items()}
    previous = cv2.cvtColor(cv2.resize(source[0], work_size), cv2.COLOR_RGB2GRAY)
    for index in range(1, CHECKPOINTS[-1] + 1):
        current = cv2.cvtColor(cv2.resize(source[index], work_size), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 4, 25, 4, 7, 1.5, 0)
        for name, polygon in points.items():
            xy = np.rint(polygon).astype(np.int32)
            xy[:, 0] = np.clip(xy[:, 0], 0, work_size[0] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, work_size[1] - 1)
            polygon += flow[xy[:, 1], xy[:, 0]]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, work_size[0] - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, work_size[1] - 1)
            if index in CHECKPOINTS:
                tracked[name][index] = polygon.copy()
        previous = current
    scale = np.asarray((work_size[0], work_size[1]), np.float32)
    return {name: {frame: polygon / scale for frame, polygon in values.items()} for name, values in tracked.items()}


def crop_region(frame: np.ndarray, polygon: np.ndarray) -> torch.Tensor:
    height, width = frame.shape[:2]
    points = polygon * np.asarray((width, height), np.float32)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    pad_x, pad_y = max(2, (x1 - x0) // 12), max(2, (y1 - y0) // 12)
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    crop = frame[y0:y1, x0:x1]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = F.interpolate(tensor, size=(224, 224), mode="bicubic", align_corners=False)
    return (tensor - MEAN) / STD


def embed(model: torch.nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    batch = torch.cat(inputs).to(device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        features = model(batch)
    return F.normalize(features.float(), dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--edited-first-frame", type=Path, required=True)
    parser.add_argument("--video", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/xichong-drz-inokko/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"),
    )
    parser.add_argument(
        "--hub",
        type=Path,
        default=Path("/home/xichong-drz-inokko/.cache/torch/hub/facebookresearch_dinov2_main"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_frames(args.source)
    edited = cv2.cvtColor(cv2.imread(str(args.edited_first_frame)), cv2.COLOR_BGR2RGB)
    tracked = track_regions(source)
    outputs = {label: read_frames(Path(path)) for label, path in args.video}
    model = torch.hub.load(str(args.hub), "dinov2_vitb14_reg", source="local", weights=str(args.checkpoint))
    model = model.eval().cuda()

    records: list[dict[str, object]] = []
    for region in REGIONS:
        edit_feature = embed(model, [crop_region(edited, tracked[region][0])])[0]
        source_features = {
            frame: embed(model, [crop_region(source[frame], tracked[region][frame])])[0] for frame in CHECKPOINTS
        }
        for label, frames in outputs.items():
            output_features = embed(
                model, [crop_region(frames[frame], tracked[region][frame]) for frame in CHECKPOINTS]
            )
            for position, frame in enumerate(CHECKPOINTS):
                edit_cosine = float(output_features[position] @ edit_feature)
                source_cosine = float(output_features[position] @ source_features[frame])
                records.append(
                    {
                        "label": label,
                        "region": region,
                        "frame": frame,
                        "edited_cosine": edit_cosine,
                        "source_cosine": source_cosine,
                        "margin": edit_cosine - source_cosine,
                    }
                )

    with (args.output_dir / "dino_margins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    medians = {
        label: {
            region: float(np.median([r["margin"] for r in records if r["label"] == label and r["region"] == region]))
            for region in REGIONS
        }
        for label in outputs
    }
    (args.output_dir / "dino_summary.json").write_text(json.dumps(medians, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, region in zip(axes, REGIONS, strict=True):
        for label in outputs:
            values = [r for r in records if r["label"] == label and r["region"] == region]
            axis.plot([r["frame"] for r in values], [r["margin"] for r in values], marker="o", label=label)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{region.title()} garment")
        axis.set_xlabel("Frame")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("DINO cosine(output, edited) - cosine(output, source)")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output_dir / "dino_margins.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
