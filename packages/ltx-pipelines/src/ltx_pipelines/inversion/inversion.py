from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_pipelines.contextflow.config import BranchCondition
from ltx_pipelines.contextflow.trajectory import RectifiedFlowMidpointStepper
from ltx_pipelines.utils.helpers import modality_from_latent_state, state_with_conditionings


@dataclass(frozen=True)
class InversionResult:
    initial_noise: torch.Tensor
    timesteps: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class RectifiedFlowInverter:
    def __init__(self, *, video_tools: VideoLatentTools, stepper: RectifiedFlowMidpointStepper | None = None) -> None:
        self.video_tools = video_tools
        self.stepper = stepper or RectifiedFlowMidpointStepper()

    def invert(
        self,
        *,
        transformer: X0Model,
        source_latent: torch.Tensor,
        source_condition: BranchCondition,
        timesteps: torch.Tensor,
    ) -> InversionResult:
        inverse_timesteps = torch.flip(timesteps[:-1], dims=[0]).contiguous()
        state = self.video_tools.create_initial_state(source_latent.device, source_latent.dtype, source_latent)
        state = state_with_conditionings(state, source_condition.conditionings, self.video_tools)

        for step_index in range(len(inverse_timesteps) - 1):
            sigma = inverse_timesteps[step_index]
            sigma_next = inverse_timesteps[step_index + 1]
            midpoint_sigma = (sigma + sigma_next) * 0.5
            modality = modality_from_latent_state(state, source_condition.context, sigma)
            denoised, _ = transformer(video=modality, audio=None, perturbations=BatchedPerturbationConfig.empty(modality.latent.shape[0]))
            if denoised is None:
                raise RuntimeError("Missing denoised output during inversion")
            midpoint_latent = self.stepper.midpoint_latent(state.latent, denoised, sigma, sigma_next)
            midpoint_modality = modality_from_latent_state(replace(state, latent=midpoint_latent), source_condition.context, midpoint_sigma)
            midpoint_denoised, _ = transformer(
                video=midpoint_modality,
                audio=None,
                perturbations=BatchedPerturbationConfig.empty(midpoint_modality.latent.shape[0]),
            )
            if midpoint_denoised is None:
                raise RuntimeError("Missing midpoint denoised output during inversion")
            state = replace(
                state,
                latent=self.stepper.step(
                    sample=state.latent,
                    midpoint_denoised=midpoint_denoised,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    midpoint_sigma=midpoint_sigma,
                ),
            )

        state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(state))
        return InversionResult(initial_noise=state.latent, timesteps=timesteps, trajectory=None)
