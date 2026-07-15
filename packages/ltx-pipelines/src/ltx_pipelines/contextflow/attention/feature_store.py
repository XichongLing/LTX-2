from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StoredKV:
    k: torch.Tensor
    v: torch.Tensor
    step_index: int
    layer_index: int
    representation_stage: str


class FeatureStore:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], StoredKV] = {}

    def put(
        self,
        *,
        step_index: int,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        representation_stage: str,
    ) -> None:
        if k.shape[1] != v.shape[1]:
            raise ValueError("k and v must have matching token lengths")
        if k.shape[2] != v.shape[2]:
            raise ValueError("k and v must have matching channel sizes")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError("stored key/value tensors must be finite")
        self._cache[(step_index, layer_index)] = StoredKV(
            k=k,
            v=v,
            step_index=step_index,
            layer_index=layer_index,
            representation_stage=representation_stage,
        )

    def get(self, *, step_index: int, layer_index: int) -> StoredKV:
        key = (step_index, layer_index)
        if key not in self._cache:
            raise KeyError(f"No stored KV for step={step_index}, layer={layer_index}")
        return self._cache[key]

    def clear_step(self, step_index: int) -> None:
        for key in [key for key in self._cache if key[0] == step_index]:
            del self._cache[key]

    def clear_all(self) -> None:
        self._cache.clear()

    def is_empty(self) -> bool:
        return not self._cache
