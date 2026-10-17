from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.flowedit import FlowEditBranchCondition, FlowEditInitMode, FlowEditResult, LTXFlowEditSampler
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import cleanup_memory, get_device
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


def _ensure_video_batch(video: torch.Tensor) -> torch.Tensor:
    if video.dim() == 4:
        return video.unsqueeze(0)
    return video


def _materialize_video(chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    pieces = list(chunks)
    if not pieces:
        raise RuntimeError("Video decoder produced no chunks")
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=1)


class EditFlowPipeline:
    """Standalone LTX FlowEdit pipeline for latent/video style-transfer tests."""

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or get_device()
        self.dtype = dtype
        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.diffusion_stage = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.scheduler = LTX2Scheduler()

    def _encode_tensor(
        self,
        encoder: VideoEncoder,
        video: torch.Tensor,
        tiling_config: TilingConfig | None,
    ) -> torch.Tensor:
        batched = _ensure_video_batch(video)
        encoder_param = next(encoder.parameters())
        batched = batched.to(device=encoder_param.device, dtype=encoder_param.dtype)
        if tiling_config is not None:
            return encoder.tiled_encode(batched, tiling_config)
        return encoder(batched)

    def _prepare_initial_latent(
        self,
        *,
        init_mode: FlowEditInitMode,
        source_video: torch.Tensor | None,
        initial_video_latent: torch.Tensor | None,
        latent_shape: VideoLatentShape,
        tiling_config: TilingConfig | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if init_mode == "latent":
            if initial_video_latent is None:
                raise ValueError("initial_video_latent is required when init_mode='latent'")
            latent = initial_video_latent.to(device=self.device, dtype=self.dtype)
            if latent.shape != latent_shape.to_torch_shape():
                raise ValueError(
                    f"initial_video_latent has shape {tuple(latent.shape)}, expected {latent_shape.to_torch_shape()}"
                )
            return latent, None

        if init_mode == "encoded-video":
            if source_video is None:
                raise ValueError("source_video is required when init_mode='encoded-video'")

            def _encode(encoder: VideoEncoder) -> torch.Tensor:
                return self._encode_tensor(encoder, source_video, tiling_config)

            latent = self.image_conditioner(_encode)
            return latent, latent

        if init_mode == "gaussian":
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latent = torch.randn(
                latent_shape.to_torch_shape(),
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
            return latent, None

        raise ValueError(f"Unsupported FlowEdit init_mode: {init_mode}")

    def __call__(  # noqa: PLR0913
        self,
        *,
        source_prompt: str,
        target_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        source_video: torch.Tensor | None = None,
        initial_video_latent: torch.Tensor | None = None,
        init_mode: FlowEditInitMode = "encoded-video",
        source_negative_prompt: str = "",
        target_negative_prompt: str = "",
        source_cfg_scale: float = 2.0,
        target_cfg_scale: float = 4.5,
        num_inference_steps: int = 50,
        skip_steps: int = 4,
        refine_steps: int = 0,
        seed: int = 0,
        tiling_config: TilingConfig | None = None,
        decode_output: bool = True,
        flowedit_prediction_space: str = "x0",
        edit_strength: float = 1.0,
    ) -> FlowEditResult:
        ctx_source, ctx_source_neg, ctx_target, ctx_target_neg = self.prompt_encoder(
            [source_prompt, source_negative_prompt, target_prompt, target_negative_prompt]
        )
        cleanup_memory()

        pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate)
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
        video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), latent_shape, frame_rate)
        initial_latent, source_video_latent = self._prepare_initial_latent(
            init_mode=init_mode,
            source_video=source_video,
            initial_video_latent=initial_video_latent,
            latent_shape=latent_shape,
            tiling_config=tiling_config,
            seed=seed,
        )
        timesteps = self.scheduler.execute(steps=num_inference_steps, latent=initial_latent).to(initial_latent.device)

        sampler = LTXFlowEditSampler(
            video_tools=video_tools,
            skip_steps=skip_steps,
            refine_steps=refine_steps,
            seed=seed,
            prediction_space=flowedit_prediction_space,
            edit_strength=edit_strength,
        )
        source_condition = FlowEditBranchCondition(
            context=ctx_source.video_encoding,
            negative_context=ctx_source_neg.video_encoding,
            cfg_scale=source_cfg_scale,
        )
        target_condition = FlowEditBranchCondition(
            context=ctx_target.video_encoding,
            negative_context=ctx_target_neg.video_encoding,
            cfg_scale=target_cfg_scale,
        )
        with self.diffusion_stage.model_context(video_tools=video_tools) as transformer:
            result = sampler.sample(
                transformer=transformer,
                initial_latent=initial_latent,
                source_condition=source_condition,
                target_condition=target_condition,
                timesteps=timesteps,
            )

        edited_video = _materialize_video(self.video_decoder(result.edited_latent)) if decode_output else None
        cleanup_memory()
        return FlowEditResult(
            initial_latent=result.initial_latent,
            edited_latent=result.edited_latent,
            timesteps=result.timesteps,
            edited_video=edited_video,
            source_video_latent=source_video_latent,
        )


FlowEditPipeline = EditFlowPipeline
