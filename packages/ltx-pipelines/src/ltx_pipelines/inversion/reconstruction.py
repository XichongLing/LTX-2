from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.rf_samplers import ContextFlowRFSolver2Sampler, LTXRFEditSampler, LTXRFInversionSampler
from ltx_pipelines.inversion.schedule import contextflow_sigmas
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import cleanup_memory

RFInversionMethod = Literal["contextflow-rf-solver-2", "ltx-rf-edit-solver-2", "ltx-rf-inversion"]


@dataclass(frozen=True)
class RFReconstructionResult:
    inverted_latent: torch.Tensor
    reconstructed_latent: torch.Tensor
    reconstructed_video: torch.Tensor
    source_video_latent: torch.Tensor
    timesteps: torch.Tensor


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


class RFReconstructionPipeline:
    def __init__(
        self,
        *,
        prompt_encoder: PromptEncoder,
        image_conditioner: ImageConditioner,
        diffusion_stage: DiffusionStage,
        video_decoder: VideoDecoder,
        video_tools: VideoLatentTools,
        scheduler: LTX2Scheduler | None = None,
        method: RFInversionMethod = "ltx-rf-inversion",
        video_encoding_tiling_config: TilingConfig | None = None,
        num_inference_steps: int = 50,
        gamma: float = 0.5,
        eta: float = 0.8,
        gamma_start_step: int = 0,
        gamma_end_step: int = 5,
        eta_start_step: int = 0,
        eta_end_step: int = 15,
        gamma_trend: str = "linear_decrease",
        eta_trend: str = "constant",
        order: str = "first",
        seed: int = 0,
        use_reverse_pullback: bool | None = None,
        contextflow_post_step_clamp: bool = False,
        contextflow_first_order: bool = False,
        contextflow_inversion_first_order: bool | None = None,
        contextflow_reconstruction_first_order: bool | None = None,
        contextflow_step_parameterization: str = "velocity",
        contextflow_step_compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.prompt_encoder = prompt_encoder
        self.image_conditioner = image_conditioner
        self.diffusion_stage = diffusion_stage
        self.video_decoder = video_decoder
        self.video_tools = video_tools
        self.scheduler = scheduler or LTX2Scheduler()
        self.method = method
        self.video_encoding_tiling_config = video_encoding_tiling_config
        self.num_inference_steps = num_inference_steps
        self.gamma = gamma
        self.eta = eta
        self.gamma_start_step = gamma_start_step
        self.gamma_end_step = gamma_end_step
        self.eta_start_step = eta_start_step
        self.eta_end_step = eta_end_step
        self.gamma_trend = gamma_trend
        self.eta_trend = eta_trend
        self.order = order
        self.seed = seed
        self.use_reverse_pullback = use_reverse_pullback
        self.contextflow_post_step_clamp = contextflow_post_step_clamp
        self.contextflow_first_order = contextflow_first_order
        self.contextflow_inversion_first_order = contextflow_inversion_first_order
        self.contextflow_reconstruction_first_order = contextflow_reconstruction_first_order
        self.contextflow_step_parameterization = contextflow_step_parameterization
        self.contextflow_step_compute_dtype = contextflow_step_compute_dtype

    def _encode_tensor(self, encoder: VideoEncoder, video: torch.Tensor) -> torch.Tensor:
        batched = _ensure_video_batch(video)
        encoder_param = next(encoder.parameters())
        batched = batched.to(device=encoder_param.device, dtype=encoder_param.dtype)
        if self.video_encoding_tiling_config is not None:
            return encoder.tiled_encode(batched, self.video_encoding_tiling_config)
        return encoder(batched)

    def _build_first_frame_condition(self, frame_latent: torch.Tensor) -> VideoConditionByLatentIndex:
        return VideoConditionByLatentIndex(latent=frame_latent, strength=1.0, latent_idx=0)

    def _build_sampler(self):
        if self.method == "contextflow-rf-solver-2":
            return ContextFlowRFSolver2Sampler(
                video_tools=self.video_tools,
                eta=self.eta,
                eta_start_step=self.eta_start_step,
                eta_end_step=self.eta_end_step,
                eta_trend=self.eta_trend,
                use_reverse_pullback=False if self.use_reverse_pullback is None else self.use_reverse_pullback,
                post_step_clamp=self.contextflow_post_step_clamp,
                first_order=self.contextflow_first_order,
                inversion_first_order=self.contextflow_inversion_first_order,
                reconstruction_first_order=self.contextflow_reconstruction_first_order,
                step_parameterization=self.contextflow_step_parameterization,
                step_compute_dtype=self.contextflow_step_compute_dtype,
            )
        if self.method == "ltx-rf-edit-solver-2":
            return LTXRFEditSampler(
                video_tools=self.video_tools,
                post_step_clamp=self.contextflow_post_step_clamp,
            )
        if self.method == "ltx-rf-inversion":
            return LTXRFInversionSampler(
                video_tools=self.video_tools,
                gamma=self.gamma,
                eta=self.eta,
                gamma_start_step=self.gamma_start_step,
                gamma_end_step=self.gamma_end_step,
                eta_start_step=self.eta_start_step,
                eta_end_step=self.eta_end_step,
                gamma_trend=self.gamma_trend,
                eta_trend=self.eta_trend,
                order=self.order,
                seed=self.seed,
                use_reverse_pullback=True if self.use_reverse_pullback is None else self.use_reverse_pullback,
            )
        raise ValueError(f"Unsupported RF inversion method: {self.method}")

    def __call__(
        self,
        *,
        source_video: torch.Tensor,
        source_first_frame: torch.Tensor | None,
        prompt: str = "",
    ) -> RFReconstructionResult:
        context = self.prompt_encoder([prompt])[0].video_encoding
        cleanup_memory()

        def _encode(encoder: VideoEncoder) -> tuple[torch.Tensor, torch.Tensor | None]:
            source_video_latent = self._encode_tensor(encoder, source_video)
            source_first_frame_latent = (
                self._encode_tensor(encoder, source_first_frame) if source_first_frame is not None else None
            )
            return source_video_latent, source_first_frame_latent

        source_video_latent, source_first_frame_latent = self.image_conditioner(_encode)
        if self.method == "contextflow-rf-solver-2":
            timesteps = contextflow_sigmas(
                self.scheduler,
                steps=self.num_inference_steps,
                latent=source_video_latent,
            )
        else:
            timesteps = self.scheduler.execute(steps=self.num_inference_steps, latent=source_video_latent)
        timesteps = timesteps.to(source_video_latent.device)

        pixel_shape = VideoPixelShape(
            batch=source_video_latent.shape[0],
            frames=source_video.shape[2],
            height=source_video.shape[3],
            width=source_video.shape[4],
            fps=float(getattr(self.video_tools, "fps", 24.0)),
        )
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=source_video_latent.shape[1])
        if self.video_tools.target_shape != latent_shape:
            raise ValueError("RF reconstruction video_tools shape does not match source video shape")

        conditionings = (
            [self._build_first_frame_condition(source_first_frame_latent)]
            if source_first_frame_latent is not None
            else []
        )
        condition = BranchCondition(context=context, conditionings=conditionings)
        sampler = self._build_sampler()
        with self.diffusion_stage.model_context(video_tools=self.video_tools) as transformer:
            sampler_result = sampler.sample(
                transformer=transformer,
                source_latent=source_video_latent,
                condition=condition,
                timesteps=timesteps,
            )

        reconstructed_video = _materialize_video(self.video_decoder(sampler_result.reconstructed_latent))
        cleanup_memory()
        return RFReconstructionResult(
            inverted_latent=sampler_result.inverted_latent,
            reconstructed_latent=sampler_result.reconstructed_latent,
            reconstructed_video=reconstructed_video,
            source_video_latent=source_video_latent,
            timesteps=sampler_result.timesteps,
        )
