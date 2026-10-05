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

VAE_SPATIAL_COMPRESSION = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a two-scale Stage-1 reference-token partition from a mask.")
    parser.add_argument("--mask", required=True, help="Mask image/video/frame-dir/tensor accepted by load_file_mask().")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--stage1-width", type=int, required=True)
    parser.add_argument("--stage1-height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--inside-scale", type=int, default=4)
    parser.add_argument("--outside-scale", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inside-strength", type=float, default=1.0)
    parser.add_argument("--outside-strength", type=float, default=1.0)
    return parser.parse_args()


def _grid_size(stage1_height: int, stage1_width: int, scale: int) -> tuple[int, int]:
    divisor = scale * VAE_SPATIAL_COMPRESSION
    if stage1_height % divisor or stage1_width % divisor:
        raise ValueError(
            f"stage1 size {stage1_width}x{stage1_height} is not divisible by "
            f"scale*{VAE_SPATIAL_COMPRESSION}={divisor}"
        )
    return stage1_height // divisor, stage1_width // divisor


def _hard_tokens(mask: torch.Tensor, *, frames: int, grid_h: int, grid_w: int, fps: float) -> torch.Tensor:
    shape = VideoLatentShape(batch=1, channels=1, frames=frames, height=grid_h, width=grid_w)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps)
    return downsample_semantic_mask_to_hard_target_tokens(mask.float(), tools).reshape(frames, grid_h, grid_w)


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or (args.num_frames - 1) % 8:
        raise ValueError("num_frames must follow the causal 8k+1 layout")
    if args.inside_scale <= 0 or args.outside_scale <= 0:
        raise ValueError("inside/outside scales must be positive")
    if args.inside_scale <= args.outside_scale:
        raise ValueError("inside-scale must be coarser/larger than outside-scale for complement tiling")
    if args.inside_scale % args.outside_scale:
        raise ValueError("inside-scale must be an integer multiple of outside-scale")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if args.inside_strength < 0 or args.outside_strength < 0:
        raise ValueError("inside/outside strengths must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = load_file_mask(
        path=args.mask,
        num_frames=args.num_frames,
        height=args.stage1_height,
        width=args.stage1_width,
        device="cpu",
    )
    binary = mask >= args.threshold
    latent_frames = 1 + (args.num_frames - 1) // 8
    inside_h, inside_w = _grid_size(args.stage1_height, args.stage1_width, args.inside_scale)
    outside_h, outside_w = _grid_size(args.stage1_height, args.stage1_width, args.outside_scale)

    inside = _hard_tokens(binary.float(), frames=latent_frames, grid_h=inside_h, grid_w=inside_w, fps=args.fps)
    ratio = args.inside_scale // args.outside_scale
    inside_on_outside_grid = inside.repeat_interleave(ratio, dim=1).repeat_interleave(ratio, dim=2)
    if inside_on_outside_grid.shape != (latent_frames, outside_h, outside_w):
        raise AssertionError(
            "inside mask does not tile the outside grid: "
            f"inside_up={tuple(inside_on_outside_grid.shape)} outside={(latent_frames, outside_h, outside_w)}"
        )
    outside = ~inside_on_outside_grid

    outside_path = output_dir / "K_outside.pt"
    inside_path = output_dir / "K_inside.pt"
    torch.save({"mask": outside}, outside_path)
    torch.save({"mask": inside}, inside_path)

    preview = (binary[0, 0, 0].to(torch.uint8).numpy() * 255)
    Image.fromarray(preview, mode="L").save(output_dir / "mask_preview.png")

    config = {
        "items": [
            {
                "name": f"outside_scale{args.outside_scale}",
                "scale": args.outside_scale,
                "keep_mask": str(outside_path),
                "strength": args.outside_strength,
            },
            {
                "name": f"inside_scale{args.inside_scale}",
                "scale": args.inside_scale,
                "keep_mask": str(inside_path),
                "strength": args.inside_strength,
            },
        ],
        "log_path": str(output_dir / "ref_partition_log.json"),
    }
    (output_dir / "ref_partition_config.json").write_text(json.dumps(config, indent=2) + "\n")

    n_inside = inside.sum(dim=(1, 2)).to(torch.int64)
    n_outside = outside.sum(dim=(1, 2)).to(torch.int64)
    expected_outside = outside_h * outside_w - (ratio * ratio) * n_inside
    if not torch.equal(n_outside, expected_outside):
        raise AssertionError(
            f"partition does not tile the outside grid: outside={n_outside.tolist()} "
            f"expected={expected_outside.tolist()}"
        )

    summary = {
        "source_mask": str(Path(args.mask).expanduser().resolve()),
        "mask_policy": "load_file_mask resize_and_center_crop; threshold; causal temporal/spatial max pool",
        "threshold": args.threshold,
        "stage1_size": [args.stage1_width, args.stage1_height],
        "pixel_frames": args.num_frames,
        "latent_frames": latent_frames,
        "inside_scale": args.inside_scale,
        "outside_scale": args.outside_scale,
        "inside_shape": list(inside.shape),
        "outside_shape": list(outside.shape),
        "inside_kept_per_frame": n_inside.tolist(),
        "outside_kept_per_frame": n_outside.tolist(),
        "total_reference_tokens": int(n_inside.sum() + n_outside.sum()),
        "inside_frame0": inside[0].to(torch.int64).tolist(),
        "outside_frame0": outside[0].to(torch.int64).tolist(),
        "config": str(output_dir / "ref_partition_config.json"),
    }
    (output_dir / "partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
