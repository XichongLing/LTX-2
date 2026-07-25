from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.guidance import AttentionInputs, AttentionMetadata, SamplingStep
from ltx_pipelines.inversion.attention.ace import ContextFlowACEPolicy
from ltx_pipelines.inversion.attention.feature_store import FeatureStore


@dataclass
class NoOpAttentionController:
    def begin_step(self, step: SamplingStep) -> None:
        return None

    def record_source(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        return None

    def modify_target(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        metadata: AttentionMetadata,
    ) -> AttentionInputs:
        return AttentionInputs(q, k, v, attention_mask)

    def end_step(self, step: SamplingStep) -> None:
        return None


@dataclass
class ContextFlowAttentionController:
    feature_store: FeatureStore
    ace_policy: ContextFlowACEPolicy
    active_layers: set[int]

    def begin_step(self, step: SamplingStep) -> None:
        return None

    def record_source(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        if layer_index not in self.active_layers:
            return
        if not self.ace_policy.schedule.is_active(step_index=step.index, num_steps=step.num_steps, layer_index=layer_index):
            return
        self.feature_store.put(
            step_index=step.index,
            layer_index=layer_index,
            k=k,
            v=v,
            representation_stage=metadata.representation_stage,
        )

    def modify_target(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        metadata: AttentionMetadata,
    ) -> AttentionInputs:
        if layer_index not in self.active_layers:
            return AttentionInputs(q, k, v, attention_mask)
        if not self.ace_policy.schedule.is_active(step_index=step.index, num_steps=step.num_steps, layer_index=layer_index):
            return AttentionInputs(q, k, v, attention_mask)
        stored = self.feature_store.get(step_index=step.index, layer_index=layer_index)
        if stored.representation_stage != metadata.representation_stage:
            raise RuntimeError("Source and target K/V were captured at different representation stages")
        return self.ace_policy.apply(
            step=step,
            layer_index=layer_index,
            target_q=q,
            target_k=k,
            target_v=v,
            source_k=stored.k,
            source_v=stored.v,
            attention_mask=attention_mask,
        )

    def end_step(self, step: SamplingStep) -> None:
        self.feature_store.clear_step(step.index)
