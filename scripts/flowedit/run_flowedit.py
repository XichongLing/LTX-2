#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

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
from ltx_pipelines.editflow import EditFlowPipeline
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.media_io import decode_video_from_file, encode_video, from_vae_range, video_preprocess
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_GEMMA_ROOT = Path("/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized")
DEFAULT_SOURCE_VIDEO = Path("/home/xichong-drz-inokko/experimental/data/pexelvideos/clips/024_cinematic_292170_656042114.mp4")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "flowedit" / "outputs" / "metro_style_transfer"
DEFAULT_SOURCE_PROMPT = "A metro train moving through a modern urban station, cinematic camera motion, realistic lighting."
DEFAULT_TARGET_PROMPT = "A metro train moving through a modern urban station in watercolor painting style, cinematic camera motion."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTX FlowEdit style-transfer experiments.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--gemma-root", default=str(DEFAULT_GEMMA_ROOT))
    parser.add_argument("--source-video", default=str(DEFAULT_SOURCE_VIDEO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-prompt", default=DEFAULT_SOURCE_PROMPT)
    parser.add_argument("--target-prompt", default=DEFAULT_TARGET_PROMPT)
    parser.add_argument("--source-negative-prompt", default="")
    parser.add_argument("--target-negative-prompt", default="")
    parser.add_argument("--source-cfg", type=float, default=2.0)
    parser.add_argument("--target-cfg", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--skip-steps", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--flowedit-prediction-space", choices=("x0", "velocity"), default="x0")
    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantization", choices=("none", "fp8-cast"), default="fp8-cast")
    parser.add_argument("--prompt-offload-mode", choices=("none", "cpu", "disk"), default="none")
    parser.add_argument("--init-mode", choices=("encoded-video", "gaussian", "latent"), default="encoded-video")
    parser.add_argument("--initial-latent-path", default=None)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--use-tiled-encode", action="store_true")
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
        spatial_config=SpatialTilingConfig(args.tile_size_pixels, args.tile_overlap_pixels),
        temporal_config=TemporalTilingConfig(args.tile_size_frames, args.tile_overlap_frames),
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
    frames = from_vae_range(frames) if vae_range else frames.clamp(0, 1)
    encode_video(video=frames, fps=int(round(fps)), audio=None, output_path=str(output_path), video_chunks_number=1)


def save_first_frame(video: torch.Tensor, output_path: Path) -> None:
    frames = tensor_video_to_frames(video).detach().cpu()
    frame = from_vae_range(frames[:1])[0]
    array = (frame.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(output_path)


def quantization_policy(kind: str, checkpoint_path: str):
    if kind == "none":
        return None
    if kind == "fp8-cast":
        return QuantizationKind.FP8_CAST.to_policy(checkpoint_path)
    raise ValueError(f"Unsupported quantization kind: {kind}")


def offload_mode(name: str) -> OffloadMode:
    return {
        "none": OffloadMode.NONE,
        "cpu": OffloadMode.CPU,
        "disk": OffloadMode.DISK,
    }[name]


def load_initial_latent(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    obj = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("edited_latent", "inverted_latent", "initial_latent", "latent"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Could not find a tensor latent in {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    torch.set_grad_enabled(False)
    args = parse_args()

    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    gemma_root = str(Path(args.gemma_root).expanduser().resolve())
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
        raise ValueError("height and width must both be divisible by 64")

    source_video = None
    if args.init_mode == "encoded-video" or args.source_video:
        logging.info("Loading source video: %s", source_video_path)
        frames = decode_video_from_file(source_video_path, device=preprocess_device)
        source_video = video_preprocess(frames, height=height, width=width, dtype=preprocess_dtype, device=preprocess_device)

    initial_latent = load_initial_latent(args.initial_latent_path)
    pipeline = EditFlowPipeline(
        distilled_checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        device=device,
        quantization=quantization_policy(args.quantization, checkpoint_path),
        offload_mode=offload_mode(args.prompt_offload_mode),
        dtype=dtype,
    )

    logging.info("Running LTX FlowEdit with %d steps", args.num_steps)
    with torch.inference_mode():
        result = pipeline(
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            source_negative_prompt=args.source_negative_prompt,
            target_negative_prompt=args.target_negative_prompt,
            source_cfg_scale=args.source_cfg,
            target_cfg_scale=args.target_cfg,
            source_video=source_video,
            initial_video_latent=initial_latent,
            init_mode=args.init_mode,
            height=height,
            width=width,
            num_frames=source_video.shape[2] if source_video is not None else ref_frames,
            frame_rate=ref_fps,
            num_inference_steps=args.num_steps,
            skip_steps=args.skip_steps,
            refine_steps=args.refine_steps,
            seed=args.seed,
            tiling_config=build_tiling_config(args),
            flowedit_prediction_space=args.flowedit_prediction_space,
            edit_strength=args.edit_strength,
            decode_output=not args.no_decode,
        )

    edited_latent_path = output_dir / "edited_latent.pt"
    initial_latent_path = output_dir / "initial_latent.pt"
    metadata_path = output_dir / "run_metadata.pt"
    torch.save(result.edited_latent.detach().cpu(), edited_latent_path)
    torch.save(result.initial_latent.detach().cpu(), initial_latent_path)
    if result.source_video_latent is not None:
        torch.save(result.source_video_latent.detach().cpu(), output_dir / "source_video_latent.pt")
    if result.edited_video is not None:
        save_video_tensor(result.edited_video, output_dir / "edited_video.mp4", ref_fps)
    if source_video is not None:
        save_video_tensor(source_video, output_dir / "source_video.mp4", ref_fps, vae_range=True)
        save_first_frame(source_video, output_dir / "source_first_frame.png")

    torch.save(
        {
            "method": "ltx-flowedit",
            "source_video": source_video_path,
            "height": height,
            "width": width,
            "fps": ref_fps,
            "source_frames": ref_frames,
            "init_mode": args.init_mode,
            "initial_latent_path": args.initial_latent_path,
            "source_prompt": args.source_prompt,
            "target_prompt": args.target_prompt,
            "source_negative_prompt": args.source_negative_prompt,
            "target_negative_prompt": args.target_negative_prompt,
            "source_cfg": args.source_cfg,
            "target_cfg": args.target_cfg,
            "num_steps": args.num_steps,
            "skip_steps": args.skip_steps,
            "refine_steps": args.refine_steps,
            "flowedit_prediction_space": args.flowedit_prediction_space,
            "edit_strength": args.edit_strength,
            "seed": args.seed,
            "quantization": args.quantization,
            "prompt_offload_mode": args.prompt_offload_mode,
            "use_tiled_encode": args.use_tiled_encode,
            "timesteps": result.timesteps.detach().cpu(),
        },
        metadata_path,
    )
    logging.info("Saved edited latent to %s", edited_latent_path)
    logging.info("Saved initial latent to %s", initial_latent_path)
    if result.edited_video is not None:
        logging.info("Saved edited video to %s", output_dir / "edited_video.mp4")
    logging.info("Saved run metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
