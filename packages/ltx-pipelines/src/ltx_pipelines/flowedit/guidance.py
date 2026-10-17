from __future__ import annotations

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.flowedit.config import FlowEditBranchCondition
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent


class FlowEditBranchGuider:
    """Run one FlowEdit branch and apply branch-specific text CFG."""

    def _predict_x0(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        context: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        modality = modality_from_latent_state(state, context, sigma)
        denoised, _ = transformer(
            video=modality,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(modality.latent.shape[0]),
        )
        if denoised is None:
            raise RuntimeError("FlowEdit branch prediction did not return a video latent")
        return post_process_latent(denoised, state.denoise_mask, state.clean_latent)

    def predict_guided_x0(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: FlowEditBranchCondition,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._predict_x0(
            transformer=transformer,
            state=state,
            context=condition.context,
            sigma=sigma,
        )
        if condition.cfg_scale == 1.0:
            return cond
        if condition.negative_context is None:
            raise ValueError("negative_context is required when cfg_scale != 1.0")
        uncond = self._predict_x0(
            transformer=transformer,
            state=state,
            context=condition.negative_context,
            sigma=sigma,
        )
        return cond + (condition.cfg_scale - 1.0) * (cond - uncond)

    def predict_velocity(
        self,
        *,
        transformer: X0Model,
        state: LatentState,
        condition: FlowEditBranchCondition,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        denoised = self.predict_guided_x0(
            transformer=transformer,
            state=state,
            condition=condition,
            sigma=sigma,
        )
        return (state.latent.float() - denoised.float()) / sigma.float()
