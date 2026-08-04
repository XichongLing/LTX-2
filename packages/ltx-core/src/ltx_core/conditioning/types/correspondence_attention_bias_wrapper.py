"""Directional target-to-reference correspondence bias for conditioning tokens."""

from dataclasses import replace

import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState, VideoLatentShape


def apply_target_to_reference_correspondence_bias(  # noqa: PLR0912,PLR0913
    attention_weights: torch.Tensor | None,
    *,
    total_tokens: int,
    num_target_tokens: int,
    reference_start: int,
    target_mask: torch.Tensor,
    target_shape: VideoLatentShape,
    reference_shape: VideoLatentShape,
    logit_bias: float,
    radius: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Multiply selected attention weights so ``log(weight)`` adds a correspondence bias.

    Only the target-query/reference-key block is changed. Target-to-target,
    reference-to-target, and other conditioning blocks retain their existing
    weights.
    """
    if logit_bias < 0:
        raise ValueError(f"logit_bias must be non-negative, got {logit_bias}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if target_shape.token_count() != num_target_tokens:
        raise ValueError(f"Target shape contains {target_shape.token_count()} tokens, expected {num_target_tokens}")

    num_reference_tokens = reference_shape.token_count()
    reference_stop = reference_start + num_reference_tokens
    if reference_stop > total_tokens:
        raise ValueError(
            f"Reference token slice [{reference_start}:{reference_stop}] exceeds total token count {total_tokens}"
        )

    mask = target_mask.to(device=device, dtype=torch.float32)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2 or mask.shape[1] != num_target_tokens:
        raise ValueError(
            f"target_mask must have shape (B, {num_target_tokens}) or ({num_target_tokens},), got {tuple(mask.shape)}"
        )

    batch_size = mask.shape[0]
    if attention_weights is None:
        weights = torch.ones((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)
    else:
        weights = attention_weights.to(device=device, dtype=dtype)
        if weights.shape[1:] != (total_tokens, total_tokens):
            raise ValueError(
                f"attention_weights must end with ({total_tokens}, {total_tokens}), got {tuple(weights.shape)}"
            )
        if weights.shape[0] not in (1, batch_size):
            raise ValueError(f"attention_weights batch must be 1 or {batch_size}, got {weights.shape[0]}")
        if weights.shape[0] == 1 and batch_size > 1:
            weights = weights.expand(batch_size, -1, -1)
        weights = weights.clone()

    target_t, target_h, target_w = torch.meshgrid(
        torch.arange(target_shape.frames, device=device),
        torch.arange(target_shape.height, device=device),
        torch.arange(target_shape.width, device=device),
        indexing="ij",
    )
    target_t = target_t.flatten()
    target_h = target_h.flatten()
    target_w = target_w.flatten()

    reference_t = torch.div(target_t * reference_shape.frames, target_shape.frames, rounding_mode="floor")
    reference_h = torch.div(target_h * reference_shape.height, target_shape.height, rounding_mode="floor")
    reference_w = torch.div(target_w * reference_shape.width, target_shape.width, rounding_mode="floor")
    query_indices = torch.arange(num_target_tokens, device=device)
    boost = torch.exp(mask * float(logit_bias)).to(dtype=dtype)

    for delta_h in range(-radius, radius + 1):
        for delta_w in range(-radius, radius + 1):
            neighbor_h = reference_h + delta_h
            neighbor_w = reference_w + delta_w
            valid = (
                (neighbor_h >= 0)
                & (neighbor_h < reference_shape.height)
                & (neighbor_w >= 0)
                & (neighbor_w < reference_shape.width)
            )
            valid_queries = query_indices[valid]
            reference_indices = (
                reference_t[valid] * reference_shape.height + neighbor_h[valid]
            ) * reference_shape.width + neighbor_w[valid]
            weights[:, valid_queries, reference_start + reference_indices] *= boost[:, valid_queries]

    return weights


class ConditioningItemCorrespondenceBiasWrapper(ConditioningItem):
    """Add a masked directional logit bias from target queries to newly appended reference keys."""

    def __init__(
        self,
        conditioning: ConditioningItem,
        *,
        target_mask: torch.Tensor,
        reference_shape: VideoLatentShape,
        logit_bias: float,
        radius: int = 0,
    ):
        self.conditioning = conditioning
        self.target_mask = target_mask
        self.reference_shape = reference_shape
        self.logit_bias = logit_bias
        self.radius = radius

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: LatentTools,
    ) -> LatentState:
        reference_start = latent_state.latent.shape[1]
        new_state = self.conditioning.apply_to(latent_state, latent_tools)
        num_new_tokens = new_state.latent.shape[1] - reference_start
        expected_reference_tokens = self.reference_shape.token_count()
        if num_new_tokens != expected_reference_tokens:
            raise ValueError(
                f"Wrapped conditioning appended {num_new_tokens} tokens, "
                f"but reference_shape describes {expected_reference_tokens}"
            )

        num_target_tokens = latent_tools.target_shape.token_count()
        attention_weights = apply_target_to_reference_correspondence_bias(
            new_state.attention_mask,
            total_tokens=new_state.latent.shape[1],
            num_target_tokens=num_target_tokens,
            reference_start=reference_start,
            target_mask=self.target_mask,
            target_shape=latent_tools.target_shape,
            reference_shape=self.reference_shape,
            logit_bias=self.logit_bias,
            radius=self.radius,
            dtype=new_state.latent.dtype,
            device=new_state.latent.device,
        )
        return replace(new_state, attention_mask=attention_weights)
