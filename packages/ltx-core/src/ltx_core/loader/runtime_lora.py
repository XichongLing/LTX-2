"""Runtime-switchable LoRA residuals for single-GPU inference.

Unlike the regular loader path, this module leaves the base weights untouched.
Each adapted linear computes ``base(x) + scale * strength * B(A(x))``. This
allows two inference branches to share one (possibly FP8-cast) base model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from ltx_core.loader.primitives import LoraStateDictWithStrength


@dataclass(frozen=True)
class RuntimeLoraFactors:
    a: torch.Tensor
    b: torch.Tensor
    strength: float


class RuntimeLoraLinear(nn.Module):
    """Wrap a linear layer with one or more switchable low-rank residuals."""

    def __init__(self, base_layer: nn.Linear, factors: Sequence[RuntimeLoraFactors]) -> None:
        super().__init__()
        if not factors:
            raise ValueError("RuntimeLoraLinear requires at least one adapter")
        self.base_layer = base_layer
        self.runtime_scale = 1.0
        self._strengths = tuple(float(item.strength) for item in factors)
        for index, item in enumerate(factors):
            self.register_buffer(f"lora_a_{index}", item.a, persistent=False)
            self.register_buffer(f"lora_b_{index}", item.b, persistent=False)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(inputs)
        if self.runtime_scale == 0.0:
            return output
        residual = None
        for index, strength in enumerate(self._strengths):
            a = getattr(self, f"lora_a_{index}")
            b = getattr(self, f"lora_b_{index}")
            item = torch.nn.functional.linear(torch.nn.functional.linear(inputs, a), b)
            item = item * (self.runtime_scale * strength)
            residual = item if residual is None else residual + item
        assert residual is not None
        return output + residual.to(dtype=output.dtype)


def _runtime_lora_modules(model: nn.Module) -> Iterator[RuntimeLoraLinear]:
    for module in model.modules():
        if isinstance(module, RuntimeLoraLinear):
            yield module


def set_runtime_lora_scale(model: nn.Module, scale: float) -> None:
    scale = float(scale)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError(f"runtime LoRA scale must be finite, got {scale}")
    modules = list(_runtime_lora_modules(model))
    if not modules:
        raise RuntimeError("model has no runtime LoRA modules")
    for module in modules:
        module.runtime_scale = scale


@contextmanager
def runtime_lora_scale(model: nn.Module, scale: float) -> Iterator[None]:
    """Temporarily set every runtime adapter to *scale* and restore it."""

    modules = list(_runtime_lora_modules(model))
    if not modules:
        raise RuntimeError("model has no runtime LoRA modules")
    previous = [module.runtime_scale for module in modules]
    set_runtime_lora_scale(model, scale)
    try:
        yield
    finally:
        for module, old_scale in zip(modules, previous, strict=True):
            module.runtime_scale = old_scale


def attach_runtime_loras(  # noqa: PLR0912
    model: nn.Module, adapters: Sequence[LoraStateDictWithStrength]
) -> int:
    """Attach remapped LoRA state dicts to matching named linear modules."""

    if not adapters:
        raise ValueError("at least one runtime LoRA adapter is required")

    grouped: dict[str, list[RuntimeLoraFactors]] = defaultdict(list)
    for adapter_index, adapter in enumerate(adapters):
        state_dict = adapter.state_dict.sd
        keys = set(state_dict)
        consumed: set[str] = set()
        for key in sorted(keys):
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            b_key = f"{prefix}.lora_B.weight"
            if b_key not in state_dict:
                raise ValueError(f"runtime LoRA adapter {adapter_index} is missing {b_key}")
            grouped[prefix].append(
                RuntimeLoraFactors(a=state_dict[key], b=state_dict[b_key], strength=adapter.strength)
            )
            consumed.update((key, b_key))
        unsupported = keys - consumed
        if unsupported:
            preview = ", ".join(sorted(unsupported)[:5])
            raise ValueError(f"runtime LoRA adapter {adapter_index} has unsupported keys: {preview}")

    named_modules = dict(model.named_modules())
    for module_name, factors in grouped.items():
        module = named_modules.get(module_name)
        if module is None:
            raise ValueError(f"runtime LoRA target module does not exist: {module_name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"runtime LoRA target {module_name} is not linear: {type(module).__name__}")
        for item in factors:
            if item.a.dim() != 2 or item.b.dim() != 2:
                raise ValueError(f"runtime LoRA tensors for {module_name} must be matrices")
            rank, in_features = item.a.shape
            out_features, b_rank = item.b.shape
            if (in_features, out_features, rank) != (module.in_features, module.out_features, b_rank):
                raise ValueError(
                    f"runtime LoRA shape mismatch for {module_name}: A={tuple(item.a.shape)}, "
                    f"B={tuple(item.b.shape)}, linear=({module.out_features}, {module.in_features})"
                )

        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = RuntimeLoraLinear(module, factors)

    if not grouped:
        raise ValueError("runtime LoRA adapters contain no A/B pairs")
    return len(grouped)
