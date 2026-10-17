from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from ltx_core.conditioning import ConditioningItem


FlowEditInitMode = Literal["encoded-video", "gaussian", "latent"]


@dataclass(frozen=True)
class FlowEditBranchCondition:
    context: torch.Tensor
    negative_context: torch.Tensor | None = None
    cfg_scale: float = 1.0
    conditionings: list[ConditioningItem] = field(default_factory=list)


@dataclass(frozen=True)
class FlowEditResult:
    initial_latent: torch.Tensor
    edited_latent: torch.Tensor
    timesteps: torch.Tensor
    edited_video: torch.Tensor | None = None
    source_video_latent: torch.Tensor | None = None
