from __future__ import annotations

from dataclasses import replace

import torch
from tqdm import tqdm

from ltx_core.model.transformer import X0Model
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.flowedit.config import FlowEditBranchCondition, FlowEditResult
from ltx_pipelines.flowedit.guidance import FlowEditBranchGuider
from ltx_pipelines.utils.helpers import state_with_conditionings


class LTXFlowEditSampler:
    """Native port of the ComfyUI LTX FlowEdit update loop.

    ComfyUI's shared sampling_function returns denoised x0 predictions. The
    default therefore applies FlowEdit in x0 space, with a velocity-space option
    kept as an explicit ablation.
    """

    def __init__(
        self,
        *,
        video_tools: VideoLatentTools,
        skip_steps: int = 4,
        refine_steps: int = 0,
        seed: int = 0,
        guider: FlowEditBranchGuider | None = None,
        prediction_space: str = "x0",
        edit_strength: float = 1.0,
    ) -> None:
        if skip_steps < 0:
            raise ValueError(f"skip_steps must be non-negative, got {skip_steps}")
        if refine_steps < 0:
            raise ValueError(f"refine_steps must be non-negative, got {refine_steps}")
        if prediction_space not in {"x0", "velocity"}:
            raise ValueError(f"Unsupported FlowEdit prediction_space: {prediction_space}")
        if edit_strength < 0:
            raise ValueError(f"edit_strength must be non-negative, got {edit_strength}")
        self.video_tools = video_tools
        self.skip_steps = skip_steps
        self.refine_steps = refine_steps
        self.seed = seed
        self.guider = guider or FlowEditBranchGuider()
        self.prediction_space = prediction_space
        self.edit_strength = edit_strength

    def _condition_state(self, state: LatentState, condition: FlowEditBranchCondition) -> LatentState:
        return state_with_conditionings(state, condition.conditionings, self.video_tools)

    def _predict_branch_value(
        self,
        *,
        transformer: X0Model,
        base_state: LatentState,
        latent_tokens: torch.Tensor,
        condition: FlowEditBranchCondition,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, LatentState]:
        state = replace(
            base_state,
            latent=latent_tokens.to(base_state.latent.dtype),
            clean_latent=latent_tokens.to(base_state.clean_latent.dtype),
        )
        state = self._condition_state(state, condition)
        denoised = self.guider.predict_guided_x0(
            transformer=transformer,
            state=state,
            condition=condition,
            sigma=sigma,
        )
        if self.prediction_space == "velocity":
            branch_value = (state.latent.float() - denoised.float()) / sigma.float()
        else:
            branch_value = denoised.float()
        state_without_conditioning = self.video_tools.clear_conditioning(state)
        value_without_conditioning = self.video_tools.clear_conditioning(replace(state, latent=branch_value)).latent
        return value_without_conditioning, state_without_conditioning

    def sample(
        self,
        *,
        transformer: X0Model,
        initial_latent: torch.Tensor,
        source_condition: FlowEditBranchCondition,
        target_condition: FlowEditBranchCondition,
        timesteps: torch.Tensor,
    ) -> FlowEditResult:
        if len(timesteps) < 2:
            raise ValueError("FlowEdit requires at least two sigma values")
        if self.skip_steps >= len(timesteps) - 1:
            raise ValueError(
                f"skip_steps={self.skip_steps} leaves no FlowEdit updates for {len(timesteps) - 1} steps"
            )

        timesteps = timesteps[self.skip_steps :]
        num_steps = len(timesteps) - 1
        if self.refine_steps > num_steps:
            raise ValueError(f"refine_steps={self.refine_steps} exceeds available FlowEdit steps={num_steps}")

        generator = torch.Generator(device=initial_latent.device).manual_seed(self.seed)
        base_state = self.video_tools.create_initial_state(initial_latent.device, initial_latent.dtype, initial_latent)
        initial_tokens = base_state.latent
        target_tokens = initial_tokens.clone()
        update_mask = base_state.denoise_mask.to(target_tokens.dtype)

        for step_index in tqdm(range(num_steps)):
            sigma = timesteps[step_index].to(device=initial_latent.device)
            sigma_next = timesteps[step_index + 1].to(device=initial_latent.device)
            noise_tokens = torch.randn(
                initial_tokens.shape,
                generator=generator,
                device=initial_tokens.device,
                dtype=initial_tokens.dtype,
            )
            source_noised_tokens = (1.0 - sigma) * initial_tokens + sigma * noise_tokens

            if step_index < num_steps - self.refine_steps:
                target_noised_tokens = target_tokens + (source_noised_tokens - initial_tokens) * update_mask
                source_velocity, _ = self._predict_branch_value(
                    transformer=transformer,
                    base_state=base_state,
                    latent_tokens=source_noised_tokens,
                    condition=source_condition,
                    sigma=sigma,
                )
            else:
                if step_index == num_steps - self.refine_steps:
                    target_tokens = target_tokens + (source_noised_tokens - initial_tokens) * update_mask
                target_noised_tokens = target_tokens
                source_velocity = torch.zeros_like(target_tokens, dtype=torch.float32)

            target_velocity, target_state = self._predict_branch_value(
                transformer=transformer,
                base_state=base_state,
                latent_tokens=target_noised_tokens,
                condition=target_condition,
                sigma=sigma,
            )
            delta_value = (target_velocity - source_velocity) * target_state.denoise_mask.float()
            target_tokens = target_tokens.float() + self.edit_strength * (sigma_next.float() - sigma.float()) * delta_value
            target_tokens = target_tokens.to(initial_tokens.dtype)

        edited_state = self.video_tools.unpatchify(replace(base_state, latent=target_tokens))
        return FlowEditResult(
            initial_latent=initial_latent,
            edited_latent=edited_state.latent,
            timesteps=timesteps,
        )
