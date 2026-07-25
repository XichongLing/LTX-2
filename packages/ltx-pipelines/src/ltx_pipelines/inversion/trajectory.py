from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.guidance import AttentionHookContext, SamplingStep
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.inversion.attention.controller import NoOpAttentionController
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent, state_with_conditionings


class RectifiedFlowMidpointStepper:
    def __init__(self, *, parameterization: str = "velocity", compute_dtype: torch.dtype = torch.float32) -> None:
        if parameterization not in {"velocity", "x0"}:
            raise ValueError(f"Unsupported step parameterization: {parameterization}")
        self.parameterization = parameterization
        self.compute_dtype = compute_dtype

    def midpoint_latent(self, sample: torch.Tensor, denoised: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        x = sample.to(self.compute_dtype)
        d = denoised.to(self.compute_dtype)
        sigma = sigma.to(self.compute_dtype)
        sigma_mid = (sigma + sigma_next.to(self.compute_dtype)) * 0.5
        if self.parameterization == "x0":
            r = sigma_mid / sigma
            return (r * x + (1 - r) * d).to(sample.dtype)
        velocity = (x - d) / sigma
        return (x + velocity * (sigma_mid - sigma)).to(sample.dtype)

    def step(
        self,
        *,
        sample: torch.Tensor,
        midpoint_sample: torch.Tensor,
        midpoint_denoised: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        midpoint_sigma: torch.Tensor,
    ) -> torch.Tensor:
        x = sample.to(self.compute_dtype)
        x_mid = midpoint_sample.to(self.compute_dtype)
        d_mid = midpoint_denoised.to(self.compute_dtype)
        dt = (sigma_next - sigma).to(self.compute_dtype)
        sigma_mid = midpoint_sigma.to(self.compute_dtype)
        if self.parameterization == "x0":
            return (x + (dt / sigma_mid) * (x_mid - d_mid)).to(sample.dtype)
        midpoint_velocity = (x_mid - d_mid) / sigma_mid
        return (x + midpoint_velocity * dt).to(sample.dtype)


class DualTrajectorySampler:
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

    def _post_step_latent(self, state: LatentState, latent: torch.Tensor) -> torch.Tensor:
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

    def _modality(
        self,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
        hook_context: AttentionHookContext | None,
    ) -> Modality:
        modality = modality_from_latent_state(state, condition.context, sigma)
        if hook_context is None:
            return modality
        return replace(modality, attention_hook_context=hook_context)

    def _predict(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: BranchCondition,
        sigma: torch.Tensor,
        step: SamplingStep,
        branch: str,
        controller: object,
    ) -> torch.Tensor:
        hook_context = AttentionHookContext(controller=controller, step=step, branch=branch)
        modality = self._modality(state, condition, sigma, hook_context)
        denoised, _ = transformer(video=modality, audio=None, perturbations=BatchedPerturbationConfig.empty(modality.latent.shape[0]))
        if denoised is None:
            raise RuntimeError(f"Missing denoised output for {branch} branch")
        return post_process_latent(denoised, state.denoise_mask, state.clean_latent)

    def sample(
        self,
        *,
        transformer: X0Model,
        initial_noise: torch.Tensor,
        source_condition: BranchCondition,
        target_condition: BranchCondition,
        timesteps: torch.Tensor,
        attention_controller: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        controller = attention_controller or NoOpAttentionController()
        no_op_controller = NoOpAttentionController()
        source_state = self.video_tools.create_initial_state(initial_noise.device, initial_noise.dtype, initial_noise.clone())
        target_state_no_ace = self.video_tools.create_initial_state(initial_noise.device, initial_noise.dtype, initial_noise.clone())
        target_state_with_ace = self.video_tools.create_initial_state(initial_noise.device, initial_noise.dtype, initial_noise.clone())
        source_state = state_with_conditionings(source_state, source_condition.conditionings, self.video_tools)
        target_state_no_ace = state_with_conditionings(target_state_no_ace, target_condition.conditionings, self.video_tools)
        target_state_with_ace = state_with_conditionings(target_state_with_ace, target_condition.conditionings, self.video_tools)

        for step_index in range(len(timesteps) - 1):
            sigma = timesteps[step_index]
            sigma_next = timesteps[step_index + 1]
            midpoint_sigma = (sigma + sigma_next) * 0.5
            step = SamplingStep(index=step_index, timestep=sigma, num_steps=len(timesteps) - 1)
            controller.begin_step(step)
            try:
                source_pred = self._predict(
                    transformer=transformer,
                    state=source_state,
                    condition=source_condition,
                    sigma=sigma,
                    step=step,
                    branch="source",
                    controller=controller,
                )
                target_pred_no_ace = self._predict(
                    transformer=transformer,
                    state=target_state_no_ace,
                    condition=target_condition,
                    sigma=sigma,
                    step=step,
                    branch="target_no_ace",
                    controller=no_op_controller,
                )
                target_pred_with_ace = self._predict(
                    transformer=transformer,
                    state=target_state_with_ace,
                    condition=target_condition,
                    sigma=sigma,
                    step=step,
                    branch="target",
                    controller=controller,
                )
                if self.first_order:
                    next_source_latent = self._first_order_step(
                        sample=source_state.latent,
                        denoised=source_pred,
                        sigma=sigma,
                        sigma_next=sigma_next,
                    )
                    next_target_latent_no_ace = self._first_order_step(
                        sample=target_state_no_ace.latent,
                        denoised=target_pred_no_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                    )
                    next_target_latent_with_ace = self._first_order_step(
                        sample=target_state_with_ace.latent,
                        denoised=target_pred_with_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                    )
                else:
                    source_mid = self.stepper.midpoint_latent(source_state.latent, source_pred, sigma, sigma_next)
                    target_mid_no_ace = self.stepper.midpoint_latent(target_state_no_ace.latent, target_pred_no_ace, sigma, sigma_next)
                    target_mid_with_ace = self.stepper.midpoint_latent(target_state_with_ace.latent, target_pred_with_ace, sigma, sigma_next)
                    source_mid_state = replace(source_state, latent=source_mid)
                    target_mid_state_no_ace = replace(target_state_no_ace, latent=target_mid_no_ace)
                    target_mid_state_with_ace = replace(target_state_with_ace, latent=target_mid_with_ace)
                    source_mid_pred = self._predict(
                        transformer=transformer,
                        state=source_mid_state,
                        condition=source_condition,
                        sigma=midpoint_sigma,
                        step=step,
                        branch="source",
                        controller=controller,
                    )
                    target_mid_pred_no_ace = self._predict(
                        transformer=transformer,
                        state=target_mid_state_no_ace,
                        condition=target_condition,
                        sigma=midpoint_sigma,
                        step=step,
                        branch="target_no_ace",
                        controller=no_op_controller,
                    )
                    target_mid_pred_with_ace = self._predict(
                        transformer=transformer,
                        state=target_mid_state_with_ace,
                        condition=target_condition,
                        sigma=midpoint_sigma,
                        step=step,
                        branch="target",
                        controller=controller,
                    )
                    next_source_latent = self.stepper.step(
                        sample=source_state.latent,
                        midpoint_sample=source_mid,
                        midpoint_denoised=source_mid_pred,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    )
                    next_target_latent_no_ace = self.stepper.step(
                        sample=target_state_no_ace.latent,
                        midpoint_sample=target_mid_no_ace,
                        midpoint_denoised=target_mid_pred_no_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    )
                    next_target_latent_with_ace = self.stepper.step(
                        sample=target_state_with_ace.latent,
                        midpoint_sample=target_mid_with_ace,
                        midpoint_denoised=target_mid_pred_with_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    )
                source_state = replace(source_state, latent=self._post_step_latent(source_state, next_source_latent))
                target_state_no_ace = replace(
                    target_state_no_ace,
                    latent=self._post_step_latent(target_state_no_ace, next_target_latent_no_ace),
                )
                target_state_with_ace = replace(
                    target_state_with_ace,
                    latent=self._post_step_latent(target_state_with_ace, next_target_latent_with_ace),
                )
            finally:
                controller.end_step(step)

        source_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(source_state))
        target_state_no_ace = self.video_tools.unpatchify(self.video_tools.clear_conditioning(target_state_no_ace))
        target_state_with_ace = self.video_tools.unpatchify(self.video_tools.clear_conditioning(target_state_with_ace))
        return source_state.latent, target_state_no_ace.latent, target_state_with_ace.latent
