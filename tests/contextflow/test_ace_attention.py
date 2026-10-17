import torch

from ltx_core.guidance import SamplingStep
from ltx_pipelines.contextflow.attention.ace import ContextFlowACEPolicy, extend_attention_mask
from ltx_pipelines.contextflow.schedule import LayerStepSchedule


def test_ace_augments_key_value_length() -> None:
    policy = ContextFlowACEPolicy(schedule=LayerStepSchedule(active_layers=frozenset({0}), start_fraction=0.0, end_fraction=1.0))
    q = torch.randn(1, 8, 16)
    tk = torch.randn(1, 8, 16)
    tv = torch.randn(1, 8, 16)
    sk = torch.randn(1, 8, 16)
    sv = torch.randn(1, 8, 16)
    result = policy.apply(
        step=SamplingStep(index=0, timestep=torch.tensor(1.0), num_steps=4),
        layer_index=0,
        target_q=q,
        target_k=tk,
        target_v=tv,
        source_k=sk,
        source_v=sv,
        attention_mask=None,
    )
    assert result.q.shape[1] == 8
    assert result.k.shape[1] == 16
    assert result.v.shape[1] == 16


def test_mask_extension_additive() -> None:
    mask = torch.zeros(1, 1, 8, 8)
    extended = extend_attention_mask(mask, added_key_tokens=8)
    assert extended.shape == (1, 1, 8, 16)
    assert torch.allclose(extended[..., 8:], torch.zeros(1, 1, 8, 8))


def test_mask_extension_boolean() -> None:
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    extended = extend_attention_mask(mask, added_key_tokens=4)
    assert extended.shape == (1, 8, 12)
    assert extended[..., -4:].all()
