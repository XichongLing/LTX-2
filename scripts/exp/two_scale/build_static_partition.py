#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
for package in (REPO_ROOT / "packages/ltx-core/src", REPO_ROOT / "packages/ltx-pipelines/src"):
    sys.path.insert(0, str(package))

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.correspondence_mask import load_file_mask
from ltx_pipelines.stage2_routing import downsample_semantic_mask_to_hard_target_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static two-scale Stage-1 reference partition.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--stage1-width", type=int, default=640)
    parser.add_argument("--stage1-height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale2-strength", type=float, default=1.0)
    parser.add_argument("--scale4-strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or (args.num_frames - 1) % 8:
        raise ValueError("num_frames must follow the causal 8k+1 layout")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = load_file_mask(
        path=args.mask,
        num_frames=args.num_frames,
        height=args.stage1_height,
        width=args.stage1_width,
        device="cpu",
    )
    mask = mask >= args.threshold
    latent_frames = 1 + (args.num_frames - 1) // 8
    scale4_shape = VideoLatentShape(batch=1, channels=1, frames=latent_frames, height=3, width=5)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), scale4_shape, args.fps)
    k4 = downsample_semantic_mask_to_hard_target_tokens(mask.float(), tools).reshape(latent_frames, 3, 5)
    k2 = ~k4.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    n4 = k4.sum(dim=(1, 2)).to(torch.int64)
    n2 = k2.sum(dim=(1, 2)).to(torch.int64)
    expected_n2 = 60 - 4 * n4
    if not torch.equal(n2, expected_n2):
        raise AssertionError(f"partition does not tile the fine grid: n2={n2.tolist()} expected={expected_n2.tolist()}")

    k2_path = output_dir / "K2.pt"
    k4_path = output_dir / "K4.pt"
    torch.save({"mask": k2}, k2_path)
    torch.save({"mask": k4}, k4_path)

    transformed = (mask[0, 0, 0].to(torch.uint8).numpy() * 255)
    Image.fromarray(transformed, mode="L").save(output_dir / "static_mask_stage1.png")

    config = {
        "items": [
            {"name": "scale2", "scale": 2, "keep_mask": str(k2_path), "strength": args.scale2_strength},
            {"name": "scale4", "scale": 4, "keep_mask": str(k4_path), "strength": args.scale4_strength},
        ],
        "log_path": str(output_dir / "ref_partition_log.json"),
    }
    (output_dir / "ref_partition_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "source_mask": str(Path(args.mask).expanduser().resolve()),
        "mask_policy": "static-repeat; exact pipeline resize_and_center_crop; threshold; spatial/temporal max pool",
        "threshold": args.threshold,
        "stage1_size": [args.stage1_width, args.stage1_height],
        "pixel_frames": args.num_frames,
        "latent_frames": latent_frames,
        "k4_shape": list(k4.shape),
        "k2_shape": list(k2.shape),
        "n4_per_frame": n4.tolist(),
        "n2_per_frame": n2.tolist(),
        "total_reference_tokens": int(n4.sum() + n2.sum()),
        "k4_frame0": k4[0].to(torch.int64).tolist(),
    }
    (output_dir / "partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
