import torch
from torch.nn.attention import SDPBackend

from ltx_core.guidance import AttentionHookContext, SamplingStep
from ltx_core.model.transformer.attention import Attention, AttentionOps, PytorchAttention
from ltx_pipelines.inversion.attention.controller import NoOpAttentionController


def _attention_module() -> Attention:
    ops = AttentionOps(
        attention_function=PytorchAttention(priority=[SDPBackend.MATH]),
        masked_attention_function=PytorchAttention(priority=[SDPBackend.MATH]),
    )
    return Attention(query_dim=8, heads=2, dim_head=4, ops=ops)


def test_controller_disabled_matches_original() -> None:
    torch.manual_seed(0)
    attn = _attention_module()
    x = torch.randn(1, 5, 8)
    baseline = attn(x)
    step = SamplingStep(index=0, timestep=torch.tensor(1.0), num_steps=1)
    hook = AttentionHookContext(controller=NoOpAttentionController(), step=step, branch="target")
    hooked = attn(x, attention_hook_context=hook, layer_index=0)
    assert torch.allclose(baseline, hooked, atol=1e-6, rtol=1e-6)


def test_source_branch_noop_matches_original() -> None:
    torch.manual_seed(0)
    attn = _attention_module()
    x = torch.randn(1, 5, 8)
    baseline = attn(x)
    step = SamplingStep(index=0, timestep=torch.tensor(1.0), num_steps=1)
    hook = AttentionHookContext(controller=NoOpAttentionController(), step=step, branch="source")
    hooked = attn(x, attention_hook_context=hook, layer_index=0)
    assert torch.allclose(baseline, hooked, atol=1e-6, rtol=1e-6)
