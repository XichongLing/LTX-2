from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
