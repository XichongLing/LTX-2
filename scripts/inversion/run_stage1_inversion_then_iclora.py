#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
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
from ltx_pipelines.inversion.attention import NoOpAttentionController
from ltx_pipelines.inversion.instrumentation import ContextFlowInstrumentation
from ltx_pipelines.inversion.inversion import RectifiedFlowInverter
from ltx_pipelines.inversion.pipeline import ContextFlowPipeline
from ltx_pipelines.inversion.trajectory import DualTrajectorySampler
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.media_io import decode_video_from_file, encode_video, from_vae_range, load_image_and_preprocess, video_preprocess
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / 'models' / 'ltx-2.3-22b-distilled-1.1.safetensors'
DEFAULT_UPSAMPLER_PATH = REPO_ROOT / 'models' / 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors'
DEFAULT_GEMMA_ROOT = Path('/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized')
DEFAULT_IC_LORA_PATH = REPO_ROOT / 'models' / 'ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors'
DEFAULT_REFERENCE_IMAGE = Path('/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs/contextflow_plane_10steps_blend_0.5/reference_first_frame.png')
DEFAULT_REFERENCE_VIDEO = Path('/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs/contextflow_plane_10steps_blend_0.5/source_video.mp4')
DEFAULT_OUTPUT_DIR = Path('/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs/contextflow_plane_stage1_inversion_to_iclora')
DEFAULT_PROMPT = (
    'Follow the motion, timing, camera movement, and scene layout of the Sintel source sequence, '
    'but render the result in the appearance, color, materials, lighting, and style implied by the edited reference image. '
    'Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB control video for structure.'
)
DEFAULT_NEGATIVE_PROMPT = 'flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Invert a reference video at IC-LoRA Stage 1 resolution, reconstruct it, then run IC-LoRA style transfer from the saved inverted latent.')
    parser.add_argument('--reference-first-frame', default=str(DEFAULT_REFERENCE_IMAGE))
    parser.add_argument('--reference-video', default=str(DEFAULT_REFERENCE_VIDEO))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--checkpoint-path', default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument('--spatial-upsampler-path', default=str(DEFAULT_UPSAMPLER_PATH))
    parser.add_argument('--gemma-root', default=str(DEFAULT_GEMMA_ROOT))
    parser.add_argument('--ic-lora-path', default=str(DEFAULT_IC_LORA_PATH))
    parser.add_argument('--num-inversion-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--negative-prompt', default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument('--image-strength', type=float, default=1.0)
    parser.add_argument('--image-crf', type=int, default=33)
    parser.add_argument('--video-strength', type=float, default=1.0)
    parser.add_argument('--conditioning-attention-strength', type=float, default=1.0)
    parser.add_argument('--quantization', default='fp8-cast')
    parser.add_argument('--skip-stage-2', action='store_true')
    parser.add_argument('--use-tiled-video-encoding', action='store_true', default=True)
    return parser.parse_args()


def read_video_metadata(video_path: str) -> tuple[int, int, int, float]:
    path = Path(video_path).expanduser().resolve()
    with av.open(str(path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == 'video'), None)
        if video_stream is None:
            raise ValueError(f'No video stream found in {path}')
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
        raise ValueError(f'Could not determine frame count for {path}')
    return width, height, frames, fps


def round_down_to_multiple(value: int, divisor: int) -> int:
    rounded = value - (value % divisor)
    if rounded <= 0:
        raise ValueError(f'Value {value} is too small to round down to a positive multiple of {divisor}')
    return rounded


def build_video_encoding_tiling_config(enabled: bool) -> TilingConfig | None:
    if not enabled:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
    )


def tensor_video_to_frames(video: torch.Tensor) -> torch.Tensor:
    if video.dim() == 4 and video.shape[-1] == 3:
        return video
    if video.dim() == 5 and video.shape[-1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f'Expected batch size 1 for saving, got shape {tuple(video.shape)}')
        return video.squeeze(0)
    if video.dim() == 5 and video.shape[1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f'Expected batch size 1 for saving, got shape {tuple(video.shape)}')
        return video.squeeze(0).permute(1, 2, 3, 0)
    raise ValueError(f'Unsupported video tensor shape for saving: {tuple(video.shape)}')


def save_video_tensor(video: torch.Tensor, output_path: Path, fps: float, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(video).detach().cpu()
    if vae_range:
        frames = from_vae_range(frames)
    else:
        frames = frames.clamp(0, 1)
    encode_video(video=frames, fps=int(round(fps)), audio=None, output_path=str(output_path), video_chunks_number=1)


def save_image_tensor(image: torch.Tensor, output_path: Path, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(image).detach().cpu()
    if vae_range:
        frames = from_vae_range(frames)
    else:
        frames = frames.clamp(0, 1)
    if frames.shape[0] != 1:
        raise ValueError(f'Expected a single frame image tensor, got {frames.shape[0]} frames')
    array = (frames[0].numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(output_path)


def run_inversion(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    device = get_device()
    dtype = torch.bfloat16
    preprocess_device = torch.device('cpu')
    preprocess_dtype = torch.float32

    ref_width, ref_height, _, ref_fps = read_video_metadata(args.reference_video)
    full_width = round_down_to_multiple(ref_width, 64)
    full_height = round_down_to_multiple(ref_height, 64)
    stage1_width = full_width // 2
    stage1_height = full_height // 2

    logging.info('Full pipeline resolution: %sx%s; Stage 1 inversion resolution: %sx%s', full_width, full_height, stage1_width, stage1_height)

    video_encoding_tiling_config = build_video_encoding_tiling_config(args.use_tiled_video_encoding)
    quantization = QuantizationKind(args.quantization).to_policy(args.checkpoint_path)

    frames = decode_video_from_file(args.reference_video, device=preprocess_device)
    source_video = video_preprocess(frames, height=stage1_height, width=stage1_width, dtype=preprocess_dtype, device=preprocess_device)
    source_first_frame = source_video[:, :, :1]

    pixel_shape = VideoPixelShape(
        batch=1,
        frames=source_video.shape[2],
        height=source_video.shape[3],
        width=source_video.shape[4],
        fps=ref_fps,
    )
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=latent_shape,
        fps=ref_fps,
    )

    pipeline = ContextFlowPipeline(
        prompt_encoder=PromptEncoder(args.checkpoint_path, args.gemma_root, dtype, device, offload_mode=OffloadMode.NONE),
        image_conditioner=ImageConditioner(args.checkpoint_path, dtype, device),
        diffusion_stage=DiffusionStage(args.checkpoint_path, dtype, device, quantization=quantization),
        video_decoder=VideoDecoder(args.checkpoint_path, dtype, device),
        inverter=RectifiedFlowInverter(video_tools=video_tools),
        trajectory_sampler=DualTrajectorySampler(video_tools=video_tools),
        attention_controller=NoOpAttentionController(),
        instrumentation=ContextFlowInstrumentation(),
        video_encoding_tiling_config=video_encoding_tiling_config,
        sequential_prompt_encoding=True,
        num_inference_steps=args.num_inversion_steps,
    )

    logging.info('Running Stage 1 inversion and reconstruction')
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        result = pipeline(
            source_video=source_video,
            source_first_frame=source_first_frame,
            edited_first_frame=source_first_frame,
            source_prompt='',
            target_prompt='',
            seed=args.seed,
        )

    reconstructed_video_path = output_dir / 'reconstructed_stage1_video.mp4'
    source_video_path = output_dir / 'source_stage1_video.mp4'
    source_first_frame_path = output_dir / 'source_stage1_first_frame.png'
    latent_path = output_dir / 'stage1_inverted_latent.pt'

    save_video_tensor(result.reconstructed_video, reconstructed_video_path, ref_fps)
    save_video_tensor(source_video, source_video_path, ref_fps, vae_range=True)
    save_image_tensor(source_first_frame, source_first_frame_path, vae_range=True)
    torch.save(
        {
            'latent': result.inverted_latent.detach().cpu(),
            'seed': args.seed,
            'num_inversion_steps': args.num_inversion_steps,
            'reference_video': str(Path(args.reference_video).expanduser().resolve()),
            'stage1_pixel_height': stage1_height,
            'stage1_pixel_width': stage1_width,
            'full_pixel_height': full_height,
            'full_pixel_width': full_width,
            'fps': ref_fps,
        },
        latent_path,
    )
    logging.info('Saved reconstructed Stage 1 video to %s', reconstructed_video_path)
    logging.info('Saved Stage 1 inverted latent to %s', latent_path)
    return latent_path, reconstructed_video_path


def run_style_transfer(args: argparse.Namespace, latent_path: Path, output_dir: Path) -> Path:
    script_path = REPO_ROOT / 'scripts' / 'style_transfer_ltx23' / 'run_ic_lora_style_transfer.py'
    output_path = output_dir / 'stylized_output.mp4'
    cmd = [
        str(REPO_ROOT / '.venv' / 'bin' / 'python3'),
        str(script_path),
        '--distilled-checkpoint-path', str(Path(args.checkpoint_path).expanduser().resolve()),
        '--spatial-upsampler-path', str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        '--gemma-root', str(Path(args.gemma_root).expanduser().resolve()),
        '--ic-lora-path', str(Path(args.ic_lora_path).expanduser().resolve()),
        '--reference-video', str(Path(args.reference_video).expanduser().resolve()),
        '--conditioning-video', str(Path(args.reference_video).expanduser().resolve()),
        '--reference-image', str(Path(args.reference_first_frame).expanduser().resolve()),
        '--output', str(output_path),
        '--prompt', args.prompt,
        '--negative-prompt', args.negative_prompt,
        '--conditioning-mode', 'rgb',
        '--seed', str(args.seed),
        '--image-strength', str(args.image_strength),
        '--image-crf', str(args.image_crf),
        '--video-strength', str(args.video_strength),
        '--conditioning-attention-strength', str(args.conditioning_attention_strength),
        '--quiet-layer-streaming',
        '--quantization', args.quantization,
        '--initial-video-latent', str(latent_path),
    ]
    if args.skip_stage_2:
        cmd.append('--skip-stage-2')

    command_path = output_dir / 'style_transfer_command.txt'
    command_path.write_text(' '.join(subprocess.list2cmdline([part]) for part in cmd) + '\n')
    env = dict(os.environ)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    log_path = output_dir / 'style_transfer.log'
    logging.info('Launching IC-LoRA style transfer with saved Stage 1 latent')
    with log_path.open('w') as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    if process.returncode != 0:
        raise RuntimeError(f'IC-LoRA style transfer failed with exit code {process.returncode}. See {log_path}')
    logging.info('Saved style-transfer video to %s', output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path, _ = run_inversion(args, output_dir)
    run_style_transfer(args, latent_path, output_dir)


if __name__ == '__main__':
    main()
