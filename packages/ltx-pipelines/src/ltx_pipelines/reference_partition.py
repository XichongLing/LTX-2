from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ReferencePartitionItem:
    name: str
    scale: int
    keep_mask_path: Path
    strength: float = 1.0


@dataclass(frozen=True)
class Stage1ReferencePartitionConfig:
    items: tuple[ReferencePartitionItem, ...]
    log_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Stage1ReferencePartitionConfig":
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text())
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or len(raw_items) != 2:
            raise ValueError("reference partition config must contain exactly two items")
        items = []
        names = set()
        for raw in raw_items:
            name = str(raw["name"])
            scale = int(raw["scale"])
            strength = float(raw.get("strength", 1.0))
            if not name or name in names:
                raise ValueError(f"reference partition item names must be non-empty and unique: {name!r}")
            if scale < 1:
                raise ValueError(f"reference partition scale must be positive, got {scale}")
            if not math.isfinite(strength) or strength < 0:
                raise ValueError(f"reference partition strength must be finite and non-negative, got {strength}")
            mask_path = Path(raw["keep_mask"])
            if not mask_path.is_absolute():
                mask_path = config_path.parent / mask_path
            mask_path = mask_path.resolve()
            if not mask_path.is_file():
                raise FileNotFoundError(f"reference partition keep mask does not exist: {mask_path}")
            items.append(ReferencePartitionItem(name, scale, mask_path, strength))
            names.add(name)
        scales = [item.scale for item in items]
        if len(set(scales)) != len(scales):
            raise ValueError(f"reference partition item scales must be unique, got {scales}")
        if scales != sorted(scales):
            raise ValueError(f"reference partition items must be ordered from finer to coarser scale, got {scales}")
        raw_log = payload.get("log_path")
        log_path = None
        if raw_log is not None:
            log_path = Path(raw_log)
            if not log_path.is_absolute():
                log_path = config_path.parent / log_path
            log_path = log_path.resolve()
        return cls(tuple(items), log_path)


def load_reference_keep_mask(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        payload = payload.get("mask", payload.get("keep_mask", payload.get("tensor")))
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"expected a tensor keep mask at {path}")
    if payload.ndim != 3:
        raise ValueError(f"reference keep mask must have shape (T,H,W), got {tuple(payload.shape)}")
    return payload.to(dtype=torch.bool, device="cpu").contiguous()
