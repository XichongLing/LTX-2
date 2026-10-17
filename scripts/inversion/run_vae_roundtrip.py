#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from fractions import Fraction
from pathlib import Path

import av
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC_DIRS = (
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
)
for src_dir in PACKAGE_SRC_DIRS:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.media_io import decode_video_from_file, encode_video, from_vae_range, video_preprocess

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_SOURCE_VIDEO = Path(
    "/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs/contextflow_plane_10steps_blend_0.5/source_video.mp4"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "inversion" / "outputs" / "vae_roundtrip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a video with the LTX-2 VAE and immediately decode it without diffusion.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH), help="Path to the LTX-2 checkpoint.")
    parser.add_argument("--source-video", default=str(DEFAULT_SOURCE_VIDEO), help="Input video path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for roundtrip outputs.")
    parser.add_argument("--height", type=int, default=None, help="Optional target height. Defaults to source height rounded down to a multiple of 64.")
    parser.add_argument("--width", type=int, default=None, help="Optional target width. Defaults to source width rounded down to a multiple of 64.")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="Model dtype for VAE encode/decode blocks.")
    parser.add_argument("--device", default=None, help="Optional device override, for example 'cuda' or 'cpu'.")
    parser.add_argument("--use-tiled-encode", action="store_true", help="Enable tiled VAE encoding.")
    parser.add_argument("--tile-size-pixels", type=int, default=256)
    parser.add_argument("--tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--tile-size-frames", type=int, default=24)
    parser.add_argument("--tile-overlap-frames", type=int, default=16)
    return parser.parse_args()


def read_video_metadata(video_path: str) -> tuple[int, int, int, float]:
    path = Path(video_path).expanduser().resolve()
    with av.open(str(path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {path}")

        width = int(video_stream.codec_context.width or video_stream.width)
        height = int(video_stream.codec_context.height or video_stream.height)
        fps_value = video_stream.average_rate or video_stream.base_rate or Fraction(25, 1)
        fps = float(fps_value)

        frames = video_stream.frames
        if not frames or frames <= 0:
            frames = 0
            for packet in container.demux(video_stream):
                for _frame in packet.decode():
                    frames += 1

    if frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")
    return width, height, frames, fps


def round_down_to_multiple(value: int, divisor: int) -> int:
    rounded = value - (value % divisor)
    if rounded <= 0:
        raise ValueError(f"Value {value} is too small to round down to a positive multiple of {divisor}")
    return rounded


def build_tiling_config(args: argparse.Namespace) -> TilingConfig | None:
    if not args.use_tiled_encode:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.tile_size_pixels,
            tile_overlap_in_pixels=args.tile_overlap_pixels,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.tile_size_frames,
            tile_overlap_in_frames=args.tile_overlap_frames,
        ),
    )


def tensor_video_to_frames(video: torch.Tensor) -> torch.Tensor:
    if video.dim() == 4 and video.shape[-1] == 3:
        return video
    if video.dim() == 5 and video.shape[-1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(video.shape)}")
        return video.squeeze(0)
    if video.dim() == 5 and video.shape[1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(video.shape)}")
        return video.squeeze(0).permute(1, 2, 3, 0)
    raise ValueError(f"Unsupported video tensor shape for saving: {tuple(video.shape)}")


def save_video_tensor(video: torch.Tensor, output_path: Path, fps: float, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(video).detach().cpu()
    if vae_range:
        frames = from_vae_range(frames)
    else:
        frames = frames.clamp(0, 1)
    encode_video(
        video=frames,
        fps=int(round(fps)),
        audio=None,
        output_path=str(output_path),
        video_chunks_number=1,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    args = parse_args()

    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    source_video_path = str(Path(args.source_video).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    preprocess_device = torch.device("cpu")
    preprocess_dtype = torch.float32

    ref_width, ref_height, ref_frames, ref_fps = read_video_metadata(source_video_path)
    width = args.width if args.width is not None else round_down_to_multiple(ref_width, 64)
    height = args.height if args.height is not None else round_down_to_multiple(ref_height, 64)
    if width % 64 != 0 or height % 64 != 0:
        raise ValueError("height and width must both be divisible by 64.")

    tiling_config = build_tiling_config(args)

    logging.info("Loading source video: %s", source_video_path)
    frames = decode_video_from_file(source_video_path, device=preprocess_device)
    source_video = video_preprocess(frames, height=height, width=width, dtype=preprocess_dtype, device=preprocess_device)
    logging.info("Preprocessed video shape: %s", tuple(source_video.shape))

    image_conditioner = ImageConditioner(checkpoint_path, dtype, device)
    video_decoder = VideoDecoder(checkpoint_path, dtype, device)

    def _encode(encoder: torch.nn.Module) -> torch.Tensor:
        batched = source_video.to(device=device, dtype=dtype)
        if tiling_config is not None:
            return encoder.tiled_encode(batched, tiling_config)
        return encoder(batched)

    logging.info("Encoding video with VAE%s", " (tiled)" if tiling_config is not None else "")
    with torch.inference_mode():
        encoded_latent = image_conditioner(_encode)
    logging.info("Encoded latent shape: %s", tuple(encoded_latent.shape))

    latent_path = output_dir / "encoded_latent.pt"
    torch.save(
        {
            "latent": encoded_latent.detach().cpu(),
            "source_video": source_video_path,
            "height": height,
            "width": width,
            "fps": ref_fps,
            "source_frames": ref_frames,
            "use_tiled_encode": args.use_tiled_encode,
        },
        latent_path,
    )

    logging.info("Decoding latent back to video")
    with torch.inference_mode():
        decoded_video = list(video_decoder(encoded_latent))
    if not decoded_video:
        raise RuntimeError("Video decoder returned no chunks")
    reconstructed_video = decoded_video[0] if len(decoded_video) == 1 else torch.cat(decoded_video, dim=1)
    logging.info("Decoded video shape: %s", tuple(reconstructed_video.shape))

    reconstructed_video_path = output_dir / "vae_roundtrip.mp4"
    preprocessed_source_path = output_dir / "preprocessed_source.mp4"

    save_video_tensor(reconstructed_video, reconstructed_video_path, ref_fps)
    save_video_tensor(source_video, preprocessed_source_path, ref_fps, vae_range=True)

    logging.info("Saved preprocessed source video to %s", preprocessed_source_path)
    logging.info("Saved encoded latent to %s", latent_path)
    logging.info("Saved VAE roundtrip video to %s", reconstructed_video_path)


if __name__ == "__main__":
    main()
