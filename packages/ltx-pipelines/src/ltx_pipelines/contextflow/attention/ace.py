from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.guidance import AttentionInputs, SamplingStep
from ltx_pipelines.contextflow.schedule import InterventionSchedule

TOKEN_DIM = 1


def extend_attention_mask(attention_mask: torch.Tensor | None, *, added_key_tokens: int) -> torch.Tensor | None:
    if attention_mask is None or added_key_tokens == 0:
        return attention_mask

    fill_value = 0.0 if torch.is_floating_point(attention_mask) else True
    extension_shape = list(attention_mask.shape)
    extension_shape[-1] = added_key_tokens
    extension = torch.full(
        extension_shape,
        fill_value=fill_value,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, extension], dim=-1)


@dataclass
class ContextFlowACEPolicy:
    schedule: InterventionSchedule
    concat_order: str = "target_source"
    mode: str = "concat"
    include_source_k: bool = True
    include_source_v: bool = True
    blend_alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {self.blend_alpha}")

    def validate_shapes(
        self,
        *,
        target_q: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
    ) -> None:
        if target_k.shape[:-2] != source_k.shape[:-2]:
            raise ValueError("target/source K batch-prefix shapes must match")
        if target_v.shape[:-2] != source_v.shape[:-2]:
            raise ValueError("target/source V batch-prefix shapes must match")
        if target_k.shape[-1] != source_k.shape[-1]:
            raise ValueError("target/source K channel sizes must match")
        if target_v.shape[-1] != source_v.shape[-1]:
            raise ValueError("target/source V channel sizes must match")
        if target_k.shape[TOKEN_DIM] != target_v.shape[TOKEN_DIM]:
            raise ValueError("target K/V token lengths must match")
        if source_k.shape[TOKEN_DIM] != source_v.shape[TOKEN_DIM]:
            raise ValueError("source K/V token lengths must match")
        if target_q.shape[TOKEN_DIM] != target_k.shape[TOKEN_DIM]:
            raise ValueError("target Q and K token lengths must match before ACE")

    def _merge(self, target: torch.Tensor, source: torch.Tensor, include_source: bool) -> torch.Tensor:
        if not include_source:
            return target
        if self.mode == "replace":
            return source
        if self.mode == "blend":
            return self.blend_alpha * target + (1.0 - self.blend_alpha) * source
        if self.mode != "concat":
            raise ValueError(f"Unsupported ACE mode: {self.mode}")
        if self.concat_order == "target_source":
            return torch.cat([target, source], dim=TOKEN_DIM)
        return torch.cat([source, target], dim=TOKEN_DIM)

    def apply(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        target_q: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> AttentionInputs:
        if not self.schedule.is_active(step_index=step.index, num_steps=step.num_steps, layer_index=layer_index):
            return AttentionInputs(target_q, target_k, target_v, attention_mask)

        self.validate_shapes(
            target_q=target_q,
            target_k=target_k,
            target_v=target_v,
            source_k=source_k,
            source_v=source_v,
        )

        k_aug = self._merge(target_k, source_k, self.include_source_k)
        v_aug = self._merge(target_v, source_v, self.include_source_v)
        added_key_tokens = 0
        if self.mode == "concat":
            if self.include_source_k or self.include_source_v:
                added_key_tokens = source_k.shape[TOKEN_DIM]
        mask_aug = extend_attention_mask(attention_mask, added_key_tokens=added_key_tokens)
        return AttentionInputs(target_q, k_aug, v_aug, mask_aug)
