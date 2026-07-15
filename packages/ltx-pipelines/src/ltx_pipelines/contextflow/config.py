from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ltx_core.conditioning import ConditioningItem


@dataclass(frozen=True)
class InversionConfig:
    type: str = "rf_solver"
    order: int = 2
    num_steps: int = 50
    use_blank_prompt: bool = True


@dataclass(frozen=True)
class ACEConfig:
    enabled: bool = True
    layers: tuple[int, ...] = (4,)
    start_fraction: float = 0.0
    end_fraction: float = 0.5
    concat_order: str = "target_source"
    representation_stage: str = "post_rope"
    mode: str = "concat"
    include_source_k: bool = True
    include_source_v: bool = True


@dataclass(frozen=True)
class ContextFlowConfig:
    method: str = "contextflow"
    inversion: InversionConfig = field(default_factory=InversionConfig)
    ace: ACEConfig = field(default_factory=ACEConfig)
    save_reconstruction: bool = True
    tensor_statistics: bool = True
    measure_memory: bool = True


@dataclass(frozen=True)
class BranchCondition:
    context: torch.Tensor
    conditionings: list[ConditioningItem] = field(default_factory=list)
