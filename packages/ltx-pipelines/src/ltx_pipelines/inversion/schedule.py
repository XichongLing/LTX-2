from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


CONTEXTFLOW_RAW_SIGMA_MIN = 0.003 / 1.002
CONTEXTFLOW_SHIFT = 5.0
CONTEXTFLOW_SIGMA_MIN = (
    CONTEXTFLOW_SHIFT * CONTEXTFLOW_RAW_SIGMA_MIN
    / (1.0 + (CONTEXTFLOW_SHIFT - 1.0) * CONTEXTFLOW_RAW_SIGMA_MIN)
)


class SigmaScheduler(Protocol):
    def execute(self, steps: int, **kwargs: object) -> torch.Tensor: ...


def contextflow_sigmas(
    scheduler: SigmaScheduler,
    *,
    steps: int,
    latent: torch.Tensor,
    sigma_min: float = CONTEXTFLOW_SIGMA_MIN,
) -> torch.Tensor:
    """Build a symmetric non-zero sigma grid for ContextFlow inversion and reconstruction."""
    if steps < 1:
        raise ValueError("ContextFlow requires at least one solver step.")
    if not 0.0 < sigma_min < 1.0:
        raise ValueError("ContextFlow sigma_min must be in (0, 1).")

    # LTX2Scheduler appends zero after its stretched terminal. Request one
    # extra interval, then remove only zero to retain exactly `steps` updates.
    sigmas = scheduler.execute(steps=steps + 1, latent=latent, terminal=sigma_min)[:-1]
    if len(sigmas) != steps + 1:
        raise ValueError(f"Expected {steps + 1} ContextFlow sigmas, got {len(sigmas)}.")
    if not torch.isfinite(sigmas).all():
        raise ValueError("ContextFlow sigma schedule contains non-finite values.")
    if torch.any(sigmas <= 0) or torch.any(sigmas[1:] >= sigmas[:-1]):
        raise ValueError("ContextFlow sigma schedule must be positive and strictly decreasing.")
    return sigmas


class InterventionSchedule(Protocol):
    def is_active(self, *, step_index: int, num_steps: int, layer_index: int) -> bool: ...


@dataclass(frozen=True)
class LayerStepSchedule:
    active_layers: frozenset[int] | None = None
    layer_range: tuple[int, int] | None = None
    start_fraction: float = 0.0
    end_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_fraction <= 1.0):
            raise ValueError("start_fraction must be in [0, 1]")
        if not (0.0 <= self.end_fraction <= 1.0):
            raise ValueError("end_fraction must be in [0, 1]")
        if self.start_fraction > self.end_fraction:
            raise ValueError("start_fraction must be <= end_fraction")
        if self.active_layers is None and self.layer_range is None:
            object.__setattr__(self, "active_layers", frozenset())

    def _layer_active(self, layer_index: int) -> bool:
        if self.active_layers is not None and len(self.active_layers) > 0:
            return layer_index in self.active_layers
        if self.layer_range is not None:
            start, end = self.layer_range
            return start <= layer_index <= end
        return False

    def _progress(self, step_index: int, num_steps: int) -> float:
        if num_steps <= 1:
            return 0.0
        return step_index / float(num_steps - 1)

    def is_active(self, *, step_index: int, num_steps: int, layer_index: int) -> bool:
        if not self._layer_active(layer_index):
            return False
        progress = self._progress(step_index, num_steps)
        return self.start_fraction <= progress <= self.end_fraction
