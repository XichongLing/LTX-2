from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.guidance import AttentionHookContext, SamplingStep
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.contextflow.attention.controller import NoOpAttentionController
from ltx_pipelines.contextflow.config import BranchCondition
from ltx_pipelines.utils.helpers import modality_from_latent_state, state_with_conditionings


class RectifiedFlowMidpointStepper:
    def midpoint_latent(self, sample: torch.Tensor, denoised: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        velocity = (sample.to(torch.float32) - denoised.to(torch.float32)) / sigma.to(torch.float32)
        dt = (sigma_next - sigma).to(torch.float32) * 0.5
        return (sample.to(torch.float32) + velocity * dt).to(sample.dtype)

    def step(
        self,
        *,
        sample: torch.Tensor,
        midpoint_denoised: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        midpoint_sigma: torch.Tensor,
    ) -> torch.Tensor:
        midpoint_velocity = (sample.to(torch.float32) - midpoint_denoised.to(torch.float32)) / midpoint_sigma.to(torch.float32)
        dt = (sigma_next - sigma).to(torch.float32)
        return (sample.to(torch.float32) + midpoint_velocity * dt).to(sample.dtype)


class DualTrajectorySampler:
    def __init__(self, *, video_tools: VideoLatentTools, stepper: RectifiedFlowMidpointStepper | None = None) -> None:
        self.video_tools = video_tools
        self.stepper = stepper or RectifiedFlowMidpointStepper()

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
        return denoised

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
                source_state = replace(
                    source_state,
                    latent=self.stepper.step(
                        sample=source_state.latent,
                        midpoint_denoised=source_mid_pred,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    ),
                )
                target_state_no_ace = replace(
                    target_state_no_ace,
                    latent=self.stepper.step(
                        sample=target_state_no_ace.latent,
                        midpoint_denoised=target_mid_pred_no_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    ),
                )
                target_state_with_ace = replace(
                    target_state_with_ace,
                    latent=self.stepper.step(
                        sample=target_state_with_ace.latent,
                        midpoint_denoised=target_mid_pred_with_ace,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        midpoint_sigma=midpoint_sigma,
                    ),
                )
            finally:
                controller.end_step(step)

        source_state = self.video_tools.unpatchify(self.video_tools.clear_conditioning(source_state))
        target_state_no_ace = self.video_tools.unpatchify(self.video_tools.clear_conditioning(target_state_no_ace))
        target_state_with_ace = self.video_tools.unpatchify(self.video_tools.clear_conditioning(target_state_with_ace))
        return source_state.latent, target_state_no_ace.latent, target_state_with_ace.latent
