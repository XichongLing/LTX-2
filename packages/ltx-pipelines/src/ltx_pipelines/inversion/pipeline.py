from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.instrumentation import ContextFlowInstrumentation
from ltx_pipelines.inversion.inversion import RectifiedFlowInverter
from ltx_pipelines.inversion.schedule import contextflow_sigmas
from ltx_pipelines.inversion.trajectory import DualTrajectorySampler
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import cleanup_memory


@dataclass(frozen=True)
class ContextFlowResult:
    edited_video: torch.Tensor
    edited_video_no_ace: torch.Tensor
    edited_video_with_ace: torch.Tensor
    reconstructed_video: torch.Tensor | None
    inverted_latent: torch.Tensor
    metrics: dict[str, float]
    debug_artifacts: dict[str, str]


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


class ContextFlowPipeline:
    def __init__(
        self,
        *,
        prompt_encoder: PromptEncoder,
        image_conditioner: ImageConditioner,
        diffusion_stage: DiffusionStage,
        video_decoder: VideoDecoder,
        inverter: RectifiedFlowInverter,
        trajectory_sampler: DualTrajectorySampler,
        attention_controller: object,
        instrumentation: ContextFlowInstrumentation,
        scheduler: LTX2Scheduler | None = None,
        negative_prompt: str = "",
        video_encoding_tiling_config: TilingConfig | None = None,
        sequential_prompt_encoding: bool = False,
        num_inference_steps: int = 50,
    ):
        self.prompt_encoder = prompt_encoder
        self.image_conditioner = image_conditioner
        self.diffusion_stage = diffusion_stage
        self.video_decoder = video_decoder
        self.inverter = inverter
        self.trajectory_sampler = trajectory_sampler
        self.attention_controller = attention_controller
        self.instrumentation = instrumentation
        self.scheduler = scheduler or LTX2Scheduler()
        self.negative_prompt = negative_prompt
        self.video_encoding_tiling_config = video_encoding_tiling_config
        self.sequential_prompt_encoding = sequential_prompt_encoding
        self.num_inference_steps = num_inference_steps

    def _encode_tensor(self, encoder: VideoEncoder, video: torch.Tensor) -> torch.Tensor:
        batched = _ensure_video_batch(video)
        encoder_param = next(encoder.parameters())
        batched = batched.to(device=encoder_param.device, dtype=encoder_param.dtype)
        if self.video_encoding_tiling_config is not None:
            return encoder.tiled_encode(batched, self.video_encoding_tiling_config)
        return encoder(batched)

    def _encode_video_inputs(
        self,
        *,
        source_video: torch.Tensor,
        source_first_frame: torch.Tensor,
        edited_first_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _encode(encoder: VideoEncoder) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                self._encode_tensor(encoder, source_video),
                self._encode_tensor(encoder, source_first_frame),
                self._encode_tensor(encoder, edited_first_frame),
            )

        return self.image_conditioner(_encode)

    def _encode_single_prompt(self, prompt: str) -> torch.Tensor:
        return self.prompt_encoder([prompt])[0].video_encoding

    def _encode_prompt_contexts(self, source_prompt: str, target_prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sequential_prompt_encoding:
            source_context = self._encode_single_prompt(source_prompt)
            target_context = self._encode_single_prompt(target_prompt)
            blank_context = self._encode_single_prompt("")
            return source_context, target_context, blank_context

        prompt_outputs = self.prompt_encoder([source_prompt, target_prompt, ""])
        return (
            prompt_outputs[0].video_encoding,
            prompt_outputs[1].video_encoding,
            prompt_outputs[2].video_encoding,
        )

    def _build_first_frame_condition(self, frame_latent: torch.Tensor) -> VideoConditionByLatentIndex:
        return VideoConditionByLatentIndex(latent=frame_latent, strength=1.0, latent_idx=0)

    def __call__(
        self,
        *,
        source_video: torch.Tensor,
        source_first_frame: torch.Tensor,
        edited_first_frame: torch.Tensor,
        source_prompt: str,
        target_prompt: str,
        seed: int,
    ) -> ContextFlowResult:
        source_context, target_context, blank_context = self._encode_prompt_contexts(source_prompt, target_prompt)
        cleanup_memory()

        source_video_latent, source_first_frame_latent, edited_first_frame_latent = self._encode_video_inputs(
            source_video=source_video,
            source_first_frame=source_first_frame,
            edited_first_frame=edited_first_frame,
        )

        reconstruction_condition = BranchCondition(
            context=blank_context,
            conditionings=[self._build_first_frame_condition(source_first_frame_latent)],
        )
        target_condition = BranchCondition(
            context=blank_context,
            conditionings=[self._build_first_frame_condition(edited_first_frame_latent)],
        )
        inversion_condition = BranchCondition(
            context=blank_context,
            conditionings=reconstruction_condition.conditionings,
        )

        timesteps = contextflow_sigmas(
            self.scheduler,
            steps=self.num_inference_steps,
            latent=source_video_latent,
        ).to(source_video_latent.device)
        video_fps = float(getattr(self.trajectory_sampler.video_tools, "fps", 24.0))
        pixel_shape = VideoPixelShape(
            batch=source_video_latent.shape[0],
            frames=source_video.shape[2],
            height=source_video.shape[3],
            width=source_video.shape[4],
            fps=video_fps,
        )
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=source_video_latent.shape[1])
        if self.trajectory_sampler.video_tools.target_shape != latent_shape:
            raise ValueError("ContextFlow pipeline video_tools shape does not match source video shape")

        generator = torch.Generator(device=source_video_latent.device).manual_seed(seed)
        del generator
        with self.diffusion_stage.model_context(video_tools=self.trajectory_sampler.video_tools) as transformer:
            inversion_result = self.inverter.invert(
                transformer=transformer,
                source_latent=source_video_latent,
                source_condition=inversion_condition,
                timesteps=timesteps,
            )
            reconstructed_latent, edited_latent_no_ace, edited_latent_with_ace = self.trajectory_sampler.sample(
                transformer=transformer,
                initial_noise=inversion_result.initial_noise,
                source_condition=reconstruction_condition,
                target_condition=target_condition,
                timesteps=inversion_result.timesteps,
                attention_controller=self.attention_controller,
            )

        reconstructed_video = _materialize_video(self.video_decoder(reconstructed_latent))
        edited_video_no_ace = _materialize_video(self.video_decoder(edited_latent_no_ace))
        edited_video_with_ace = _materialize_video(self.video_decoder(edited_latent_with_ace))
        cleanup_memory()
        return ContextFlowResult(
            edited_video=edited_video_with_ace,
            edited_video_no_ace=edited_video_no_ace,
            edited_video_with_ace=edited_video_with_ace,
            reconstructed_video=reconstructed_video,
            inverted_latent=inversion_result.initial_noise,
            metrics={"num_attention_records": float(len(self.instrumentation.records))},
            debug_artifacts={},
        )
