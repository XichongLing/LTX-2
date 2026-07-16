from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class SamplingStep:
    index: int
    timestep: torch.Tensor
    num_steps: int


@dataclass(frozen=True)
class AttentionMetadata:
    representation_stage: str


@dataclass(frozen=True)
class AttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_mask: torch.Tensor | None


class AttentionController(Protocol):
    def begin_step(self, step: SamplingStep) -> None: ...

    def record_source(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None: ...

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
    ) -> AttentionInputs: ...

    def end_step(self, step: SamplingStep) -> None: ...


@dataclass(frozen=True)
class AttentionHookContext:
    controller: AttentionController
    step: SamplingStep
    branch: str
