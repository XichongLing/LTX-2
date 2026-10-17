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

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.inversion.reconstruction import RFReconstructionPipeline
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.media_io import decode_video_from_file, encode_video, from_vae_range, load_image_and_preprocess, video_preprocess
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_GEMMA_ROOT = Path("/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized")
# DEFAULT_SOURCE_VIDEO = Path(
#     "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/source_video.mp4"
# )
# DEFAULT_SOURCE_VIDEO = Path("/home/xichong-drz-inokko/experimental/data/Blockout/videos/fox_low.mp4")
DEFAULT_SOURCE_VIDEO = Path("/home/xichong-drz-inokko/experimental/data/pexelvideos/clips/024_cinematic_292170_656042114.mp4")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "inversion" / "outputs" / "rf_inversion_reconstruction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated RF inversion/reconstruction on LTX-2.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--gemma-root", default=str(DEFAULT_GEMMA_ROOT))
    parser.add_argument("--source-video", default=str(DEFAULT_SOURCE_VIDEO))
    parser.add_argument(
        "--first-frame",
        default="source_video_frame_0",
        help="First-frame image condition. Default uses frame 0 of --source-video; pass None/null/no to disable first-frame conditioning.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--method", choices=("contextflow-rf-solver-2", "ltx-rf-edit-solver-2", "ltx-rf-inversion"), default="ltx-rf-inversion")
    parser.add_argument("--prompt", default="", help="Prompt context for inversion/reconstruction. Empty string mimics null-prompt inversion.")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantization", choices=("none", "fp8-cast"), default="fp8-cast")
    parser.add_argument("--prompt-offload-mode", choices=("none", "cpu", "disk"), default="none")
    parser.add_argument("--use-tiled-encode", action="store_true")
    parser.add_argument("--tile-size-pixels", type=int, default=256)
    parser.add_argument("--tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--tile-size-frames", type=int, default=24)
    parser.add_argument("--tile-overlap-frames", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--gamma-start-step", type=int, default=0)
    parser.add_argument("--gamma-end-step", type=int, default=5)
    parser.add_argument("--eta-start-step", type=int, default=0)
    parser.add_argument("--eta-end-step", type=int, default=15)
    parser.add_argument("--gamma-trend", choices=("linear_decrease", "linear_increase", "constant"), default="linear_decrease")
    parser.add_argument("--eta-trend", choices=("linear_decrease", "linear_increase", "constant"), default="constant")
    parser.add_argument("--order", choices=("first", "second"), default="first")
    parser.add_argument(
        "--disable-reverse-pullback",
        action="store_true",
        help="Ablation: remove the explicit reverse pull toward the clean source latent y0.",
    )
    parser.add_argument(
        "--enable-contextflow-reverse-pullback",
        action="store_true",
        help="Ablation for contextflow-rf-solver-2: add an official-LTX-style reverse pull toward clean y0.",
    )
    parser.add_argument(
        "--enable-contextflow-post-step-clamp",
        action="store_true",
        help="Ablation for contextflow-rf-solver-2: apply latent conditioning clamp after each solver update.",
    )
    parser.add_argument(
        "--enable-contextflow-first-order",
        action="store_true",
        help="Ablation for contextflow-rf-solver-2: use first-order Euler updates for both inversion and reconstruction.",
    )
    parser.add_argument(
        "--enable-contextflow-inversion-first-order",
        action="store_true",
        help="Ablation for contextflow-rf-solver-2: use first-order Euler updates only for inversion.",
    )
    parser.add_argument(
        "--enable-contextflow-reconstruction-first-order",
        action="store_true",
        help="Ablation for contextflow-rf-solver-2: use first-order Euler updates only for reconstruction.",
    )
    parser.add_argument(
        "--contextflow-step-parameterization",
        choices=("velocity", "x0"),
        default="velocity",
        help="Ablation for contextflow-rf-solver-2: use velocity-form or algebraically equivalent x0-form arithmetic.",
    )
    parser.add_argument(
        "--contextflow-step-compute-dtype",
        choices=("bf16", "fp32", "fp64"),
        default="fp32",
        help="Compute dtype for contextflow-rf-solver-2 step arithmetic.",
    )
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


def save_image_tensor(image: torch.Tensor, output_path: Path, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(image).detach().cpu()
    frames = from_vae_range(frames) if vae_range else frames.clamp(0, 1)
    if frames.shape[0] != 1:
        raise ValueError(f"Expected a single frame image tensor, got {frames.shape[0]} frames")
    array = (frames[0].numpy() * 255.0).round().astype(np.uint8)
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


def contextflow_step_compute_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }[name]


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
        raise ValueError("height and width must both be divisible by 64.")

    logging.info("Loading source video: %s", source_video_path)
    frames = decode_video_from_file(source_video_path, device=preprocess_device)
    source_video = video_preprocess(frames, height=height, width=width, dtype=preprocess_dtype, device=preprocess_device)
    first_frame_arg = args.first_frame.strip().lower()
    first_frame_condition_enabled = first_frame_arg not in {"none", "null", "no", "false", "0"}
    source_first_frame = source_video[:, :, :1] if first_frame_condition_enabled else None
    first_frame_condition_source = "disabled"
    if first_frame_condition_enabled:
        first_frame_condition_source = "source_video_frame_0"
        if first_frame_arg not in {"", "source", "source_video_frame_0", "source-video-frame-0"}:
            first_frame_condition_source = str(Path(args.first_frame).expanduser().resolve())
            source_first_frame = load_image_and_preprocess(
                first_frame_condition_source,
                height=height,
                width=width,
                dtype=preprocess_dtype,
                device=preprocess_device,
            )
    logging.info("First-frame condition: %s", first_frame_condition_source)

    pixel_shape = VideoPixelShape(batch=1, frames=source_video.shape[2], height=height, width=width, fps=ref_fps)
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=latent_shape,
        fps=ref_fps,
    )

    pipeline = RFReconstructionPipeline(
        prompt_encoder=PromptEncoder(
            checkpoint_path,
            gemma_root,
            dtype,
            device,
            offload_mode=offload_mode(args.prompt_offload_mode),
        ),
        image_conditioner=ImageConditioner(checkpoint_path, dtype, device),
        diffusion_stage=DiffusionStage(
            checkpoint_path,
            dtype,
            device,
            quantization=quantization_policy(args.quantization, checkpoint_path),
        ),
        video_decoder=VideoDecoder(checkpoint_path, dtype, device),
        video_tools=video_tools,
        method=args.method,
        video_encoding_tiling_config=build_tiling_config(args),
        num_inference_steps=args.num_steps,
        gamma=args.gamma,
        eta=args.eta,
        gamma_start_step=args.gamma_start_step,
        gamma_end_step=args.gamma_end_step,
        eta_start_step=args.eta_start_step,
        eta_end_step=args.eta_end_step,
        gamma_trend=args.gamma_trend,
        eta_trend=args.eta_trend,
        order=args.order,
        seed=args.seed,
        use_reverse_pullback=(not args.disable_reverse_pullback) if args.method == "ltx-rf-inversion" else args.enable_contextflow_reverse_pullback,
        contextflow_post_step_clamp=args.enable_contextflow_post_step_clamp,
        contextflow_first_order=args.enable_contextflow_first_order,
        contextflow_inversion_first_order=args.enable_contextflow_first_order or args.enable_contextflow_inversion_first_order,
        contextflow_reconstruction_first_order=args.enable_contextflow_first_order or args.enable_contextflow_reconstruction_first_order,
        contextflow_step_parameterization=args.contextflow_step_parameterization,
        contextflow_step_compute_dtype=contextflow_step_compute_dtype(args.contextflow_step_compute_dtype),
    )

    logging.info("Running %s with %d steps", args.method, args.num_steps)
    with torch.inference_mode():
        result = pipeline(source_video=source_video, source_first_frame=source_first_frame, prompt=args.prompt)

    latent_path = output_dir / "inverted_latent.pt"
    reconstructed_latent_path = output_dir / "reconstructed_latent.pt"
    reconstructed_video_path = output_dir / "reconstructed_video.mp4"
    source_video_output_path = output_dir / "source_video.mp4"
    first_frame_output_path = output_dir / "reference_first_frame.png"
    metadata_path = output_dir / "run_metadata.pt"

    torch.save(result.inverted_latent.detach().cpu(), latent_path)
    torch.save(result.reconstructed_latent.detach().cpu(), reconstructed_latent_path)
    torch.save(
        {
            "method": args.method,
            "source_video": source_video_path,
            "first_frame": args.first_frame,
            "first_frame_condition_source": first_frame_condition_source,
            "first_frame_condition_enabled": first_frame_condition_enabled,
            "uses_source_video_first_frame_condition": first_frame_condition_source == "source_video_frame_0",
            "height": height,
            "width": width,
            "fps": ref_fps,
            "source_frames": ref_frames,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "gamma": args.gamma,
            "eta": args.eta,
            "gamma_start_step": args.gamma_start_step,
            "gamma_end_step": args.gamma_end_step,
            "eta_start_step": args.eta_start_step,
            "eta_end_step": args.eta_end_step,
            "gamma_trend": args.gamma_trend,
            "eta_trend": args.eta_trend,
            "order": args.order,
            "use_reverse_pullback": (not args.disable_reverse_pullback) if args.method == "ltx-rf-inversion" else args.enable_contextflow_reverse_pullback,
            "disable_reverse_pullback": args.disable_reverse_pullback,
            "enable_contextflow_reverse_pullback": args.enable_contextflow_reverse_pullback,
            "enable_contextflow_post_step_clamp": args.enable_contextflow_post_step_clamp,
            "enable_contextflow_first_order": args.enable_contextflow_first_order,
            "enable_contextflow_inversion_first_order": args.enable_contextflow_first_order or args.enable_contextflow_inversion_first_order,
            "enable_contextflow_reconstruction_first_order": args.enable_contextflow_first_order or args.enable_contextflow_reconstruction_first_order,
            "contextflow_step_parameterization": args.contextflow_step_parameterization,
            "contextflow_step_compute_dtype": args.contextflow_step_compute_dtype,
            "timesteps": result.timesteps.detach().cpu(),
        },
        metadata_path,
    )
    save_video_tensor(result.reconstructed_video, reconstructed_video_path, ref_fps)
    save_video_tensor(source_video, source_video_output_path, ref_fps, vae_range=True)
    if source_first_frame is not None:
        save_image_tensor(source_first_frame, first_frame_output_path, vae_range=True)

    logging.info("Saved inverted latent to %s", latent_path)
    logging.info("Saved reconstructed latent to %s", reconstructed_latent_path)
    logging.info("Saved reconstructed video to %s", reconstructed_video_path)
    logging.info("Saved source video to %s", source_video_output_path)
    if source_first_frame is not None:
        logging.info("Saved first-frame condition to %s", first_frame_output_path)
    else:
        logging.info("First-frame conditioning disabled; no reference_first_frame.png saved")
    logging.info("Saved run metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
