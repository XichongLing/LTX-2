from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.loader.runtime_lora import (
    RuntimeLoraFactors,
    RuntimeLoraLinear,
    attach_runtime_loras,
    runtime_lora_scale,
)
from ltx_core.quantization.fp8_cast import Fp8CastLinear


def _adapter(a: torch.Tensor, b: torch.Tensor, strength: float = 1.0) -> LoraStateDictWithStrength:
    return LoraStateDictWithStrength(
        StateDict(
            {
                "proj.lora_A.weight": a,
                "proj.lora_B.weight": b,
            },
            device=torch.device("cpu"),
            size=0,
            dtype=a.dtype,
        ),
        strength,
    )


def test_runtime_lora_scale_zero_is_exact_base_and_scale_one_matches_fusion() -> None:
    torch.manual_seed(0)
    model = nn.Module()
    model.proj = nn.Linear(4, 3, bias=False)
    base_weight = model.proj.weight.detach().clone()
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    inputs = torch.randn(5, 4)

    assert attach_runtime_loras(model, [_adapter(a, b, strength=0.25)]) == 1
    assert isinstance(model.proj, RuntimeLoraLinear)

    with runtime_lora_scale(model, 0.0):
        disabled = model.proj(inputs)
    expected_base = torch.nn.functional.linear(inputs, base_weight)
    assert torch.equal(disabled, expected_base)

    with runtime_lora_scale(model, 1.0):
        enabled = model.proj(inputs)
    fused_weight = base_weight + 0.25 * (b @ a)
    assert torch.allclose(enabled, torch.nn.functional.linear(inputs, fused_weight), atol=1e-6)


def test_runtime_lora_context_restores_scale_after_error() -> None:
    model = nn.Module()
    model.proj = nn.Linear(2, 2, bias=False)
    attach_runtime_loras(model, [_adapter(torch.eye(2), torch.eye(2))])
    model.proj.runtime_scale = 0.4

    with pytest.raises(RuntimeError, match="boom"):
        with runtime_lora_scale(model, 0.0):
            assert model.proj.runtime_scale == 0.0
            raise RuntimeError("boom")
    assert model.proj.runtime_scale == 0.4


def test_runtime_lora_rejects_missing_pair_and_shape_mismatch() -> None:
    model = nn.Module()
    model.proj = nn.Linear(4, 3, bias=False)
    missing = StateDict(
        {"proj.lora_A.weight": torch.randn(2, 4)},
        device=torch.device("cpu"),
        size=0,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="missing"):
        attach_runtime_loras(model, [LoraStateDictWithStrength(missing, 1.0)])

    with pytest.raises(ValueError, match="shape mismatch"):
        attach_runtime_loras(model, [_adapter(torch.randn(2, 5), torch.randn(3, 2))])


def test_fp8_runtime_residual_is_close_to_prefused_fp8_oracle() -> None:
    torch.manual_seed(1)
    weight = torch.randn(3, 4, dtype=torch.bfloat16) * 0.1
    a = torch.randn(2, 4, dtype=torch.bfloat16) * 0.1
    b = torch.randn(3, 2, dtype=torch.bfloat16) * 0.1
    inputs = torch.randn(20, 4, dtype=torch.bfloat16)

    base = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(weight.to(torch.float8_e4m3fn))
    base.__class__ = Fp8CastLinear
    base._with_stochastic_rounding = False
    base._seed = 0
    runtime = RuntimeLoraLinear(base, [RuntimeLoraFactors(a, b, 1.0)])

    runtime_output = runtime(inputs)
    prefused_weight = (weight.to(torch.float8_e4m3fn).to(torch.bfloat16) + b @ a).to(torch.float8_e4m3fn)
    prefused_output = torch.nn.functional.linear(inputs, prefused_weight.to(torch.bfloat16))
    assert torch.allclose(runtime_output, prefused_output, atol=0.015, rtol=0.05)
