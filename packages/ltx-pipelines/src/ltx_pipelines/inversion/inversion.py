from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.trajectory import RectifiedFlowMidpointStepper
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent, state_with_conditionings


@dataclass(frozen=True)
class InversionResult:
    initial_noise: torch.Tensor
    timesteps: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class RectifiedFlowInverter:
    def __init__(
        self,
        *,
        video_tools: VideoLatentTools,
        stepper: RectifiedFlowMidpointStepper | None = None,
        post_step_clamp: bool = False,
        first_order: bool = False,
        step_parameterization: str = "velocity",
        step_compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.video_tools = video_tools
        self.stepper = stepper or RectifiedFlowMidpointStepper(
            parameterization=step_parameterization,
            compute_dtype=step_compute_dtype,
        )
        self.post_step_clamp = post_step_clamp
        self.first_order = first_order
        self.step_parameterization = step_parameterization
        self.step_compute_dtype = step_compute_dtype

    def _post_step_latent(self, state, latent: torch.Tensor) -> torch.Tensor:
        if not self.post_step_clamp:
            return latent
        return post_process_latent(latent, state.denoise_mask, state.clean_latent)

    def _first_order_step(
        self,
        *,
        sample: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        x = sample.to(self.step_compute_dtype)
        d = denoised.to(self.step_compute_dtype)
        sigma = sigma.to(self.step_compute_dtype)
        sigma_next = sigma_next.to(self.step_compute_dtype)
        if self.step_parameterization == "x0":
            r = sigma_next / sigma
            return (r * x + (1 - r) * d).to(sample.dtype)
        velocity = (x - d) / sigma
        return (x + velocity * (sigma_next - sigma)).to(sample.dtype)

    def invert(
        self,
        *,
        transformer: X0Model,
        source_latent: torch.Tensor,
        source_condition: BranchCondition,
        timesteps: torch.Tensor,
    ) -> InversionResult:
        if len(timesteps) < 2:
            raise ValueError("ContextFlow inversion requires at least two sigma values.")
        if not torch.isfinite(timesteps).all():
            raise ValueError("ContextFlow inversion received non-finite sigma values.")
        if torch.any(timesteps <= 0):
            raise ValueError("ContextFlow inversion requires a strictly positive sigma schedule.")
        if torch.any(timesteps[1:] >= timesteps[:-1]):
            raise ValueError("ContextFlow inversion requires strictly decreasing reconstruction sigmas.")

        inverse_timesteps = torch.flip(timesteps, dims=[0]).contiguous()
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
            denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
            if self.first_order:
                next_latent = self._first_order_step(
                    sample=state.latent,
                    denoised=denoised,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )
            else:
                midpoint_latent = self.stepper.midpoint_latent(state.latent, denoised, sigma, sigma_next)
                midpoint_modality = modality_from_latent_state(replace(state, latent=midpoint_latent), source_condition.context, midpoint_sigma)
                midpoint_denoised, _ = transformer(
                    video=midpoint_modality,
                    audio=None,
                    perturbations=BatchedPerturbationConfig.empty(midpoint_modality.latent.shape[0]),
                )
                if midpoint_denoised is None:
                    raise RuntimeError("Missing midpoint denoised output during inversion")
                midpoint_denoised = post_process_latent(midpoint_denoised, state.denoise_mask, state.clean_latent)
                next_latent = self.stepper.step(
                    sample=state.latent,
                    midpoint_sample=midpoint_latent,
                    midpoint_denoised=midpoint_denoised,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    midpoint_sigma=midpoint_sigma,
                )
            state = replace(state, latent=self._post_step_latent(state, next_latent))

        state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(state))
        return InversionResult(initial_noise=state.latent, timesteps=timesteps, trajectory=None)
