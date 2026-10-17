#!/usr/bin/env python3
"""Decode the clean upscaled Stage-1 latent stored in a Stage-2 cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[3]
for package in ("ltx-core", "ltx-pipelines"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.blocks import VideoDecoder  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models/ltx-2.3-22b-distilled-1.1.safetensors")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    tensors = load_file(str(args.cache.expanduser().resolve()), device="cpu")
    latent = tensors["video_latent"].to(device="cuda", dtype=torch.bfloat16)
    tiling = TilingConfig.default()
    decoder = VideoDecoder(str(args.checkpoint.expanduser().resolve()), torch.bfloat16, torch.device("cuda"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        decoded = decoder(latent, tiling)
        encode_video(
            video=decoded,
            fps=args.fps,
            audio=None,
            output_path=str(args.output.expanduser().resolve()),
            video_chunks_number=get_video_chunks_number(args.frames, tiling),
        )


if __name__ == "__main__":
    main()
