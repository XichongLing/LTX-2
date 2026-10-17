from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ltx_pipelines.stage2_kv import Stage2KVCache
from ltx_pipelines.stage2_reference import Stage2ReferenceAttentionController


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    _heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask
    return torch.softmax(logits, dim=-1) @ v


def _controller(
    tmp_path: Path,
    *,
    mode: str = "blend",
    strength: float = 0.5,
    position_mode: str = "pre-rope",
):
    return Stage2ReferenceAttentionController(
        target_token_count=4,
        target_hard_mask=torch.tensor([[False, True, False, True]]),
        reference_token_count=3,
        reference_hard_mask=torch.tensor([[False, True, True]]),
        mode=mode,
        position_mode=position_mode,
        strength=strength,
        layer_spec="all",
        cache=Stage2KVCache(),
    )


def _run_reference_then_hybrid(
    controller: Stage2ReferenceAttentionController,
    *,
    reference_k: torch.Tensor,
    reference_v: torch.Tensor,
    hybrid_q: torch.Tensor,
    hybrid_k: torch.Tensor,
    hybrid_v: torch.Tensor,
) -> torch.Tensor:
    controller.begin_step(0)
    controller.phase = "reference"
    controller.set_pre_rope(0, torch.zeros_like(reference_k), reference_k)
    controller.attend(
        layer_index=0,
        original=_attention,
        q=reference_k,
        k=reference_k,
        v=reference_v,
        heads=1,
        mask=None,
    )
    controller.phase = "hybrid"
    controller.set_pre_rope(0, hybrid_q, hybrid_k)
    controller._masked_attention[0] = _attention
    return controller.attend(
        layer_index=0,
        original=_attention,
        q=hybrid_q,
        k=hybrid_k,
        v=hybrid_v,
        heads=1,
        mask=None,
    )


def test_reference_attention_changes_only_selected_target_queries(tmp_path: Path) -> None:
    controller = _controller(tmp_path, strength=0.75)
    reference_k = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])
    reference_v = torch.tensor([[[99.0, 99.0], [10.0, 0.0], [0.0, 10.0]]])
    hybrid_q = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    hybrid_k = torch.zeros(1, 4, 2)
    hybrid_v = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    output = _run_reference_then_hybrid(
        controller,
        reference_k=reference_k,
        reference_v=reference_v,
        hybrid_q=hybrid_q,
        hybrid_k=hybrid_k,
        hybrid_v=hybrid_v,
    )
    baseline = _attention(hybrid_q, hybrid_k, hybrid_v, 1)
    assert torch.equal(output[:, [0, 2]], baseline[:, [0, 2]])
    assert not torch.equal(output[:, [1, 3]], baseline[:, [1, 3]])
    assert controller.selected_reference_tokens == 2
    assert controller.selected_target_tokens == 2


def test_blend_strength_one_matches_replace(tmp_path: Path) -> None:
    tensors = [torch.randn(1, count, 3) for count in (3, 3, 4, 4, 4)]
    blend = _run_reference_then_hybrid(
        _controller(tmp_path, mode="blend", strength=1.0),
        reference_k=tensors[0],
        reference_v=tensors[1],
        hybrid_q=tensors[2],
        hybrid_k=tensors[3],
        hybrid_v=tensors[4],
    )
    replace = _run_reference_then_hybrid(
        _controller(tmp_path, mode="replace", strength=1.0),
        reference_k=tensors[0],
        reference_v=tensors[1],
        hybrid_q=tensors[2],
        hybrid_k=tensors[3],
        hybrid_v=tensors[4],
    )
    assert torch.equal(blend, replace)


def test_zero_strength_is_exact_noop(tmp_path: Path) -> None:
    controller = _controller(tmp_path, strength=0.0)
    q = k = v = torch.randn(1, 4, 2)
    controller.begin_step(0)
    controller.phase = "hybrid"
    output = controller.attend(layer_index=0, original=_attention, q=q, k=k, v=v, heads=1, mask=None)
    assert torch.equal(output, _attention(q, k, v, 1))
    assert controller.cache.bytes_stored == 0


def test_reference_cache_contains_only_masked_tokens(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    reference_k = torch.randn(1, 3, 4)
    reference_v = torch.randn(1, 3, 4)
    controller.begin_step(0)
    controller.phase = "reference"
    controller.set_pre_rope(0, torch.zeros_like(reference_k), reference_k)
    controller.attend(
        layer_index=0,
        original=_attention,
        q=reference_k,
        k=reference_k + 100,
        v=reference_v,
        heads=1,
        mask=None,
    )
    expected_bytes = 2 * 4 * 2 * torch.tensor([], dtype=torch.bfloat16).element_size()
    assert controller.cache.bytes_stored == expected_bytes
    cached_k, cached_v = controller.cache.consume(0, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(cached_k, reference_k[:, 1:].to(torch.bfloat16).float())
    assert torch.equal(cached_v, reference_v[:, 1:].to(torch.bfloat16).float())


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.factor


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_norm = _Scale(2.0)
        self.k_norm = _Scale(3.0)
        self.preattention_function = lambda q, k, _module, _mask, _pe, _k_pe: (q + 100, k + 100)
        self.attention_function = _attention
        self.masked_attention_function = _attention


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _FakeAttention()


def test_patch_captures_normalized_pre_rope_keys(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    transformer = nn.Module()
    transformer.velocity_model = nn.Module()
    transformer.velocity_model.transformer_blocks = nn.ModuleList([_FakeBlock()])
    attention = transformer.velocity_model.transformer_blocks[0].attn1
    raw_q = torch.ones(1, 3, 2)
    raw_k = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    values = torch.randn(1, 3, 2)

    with controller.patch_transformer(transformer):
        controller.begin_step(0)
        controller.phase = "reference"
        post_q, post_k = attention.preattention_function(raw_q, raw_k, attention, None, None, None)
        attention.attention_function(post_q, post_k, values, 1)
        cached_k, _ = controller.cache.consume(0, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(cached_k, (raw_k * 3.0)[:, 1:].to(torch.bfloat16).float())
    assert not torch.equal(cached_k, post_k[:, 1:].to(torch.bfloat16).float())


def test_empty_masks_and_invalid_replace_strength_fail_early(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selects no tokens"):
        Stage2ReferenceAttentionController(
            target_token_count=2,
            target_hard_mask=torch.zeros(1, 2, dtype=torch.bool),
            reference_token_count=3,
            reference_hard_mask=torch.ones(1, 3, dtype=torch.bool),
            mode="blend",
            strength=0.5,
            layer_spec="all",
            cache=Stage2KVCache(),
        )
    with pytest.raises(ValueError, match="requires strength=1"):
        Stage2ReferenceAttentionController(
            target_token_count=2,
            target_hard_mask=torch.ones(1, 2, dtype=torch.bool),
            reference_token_count=3,
            reference_hard_mask=torch.ones(1, 3, dtype=torch.bool),
            mode="replace",
            strength=0.5,
            layer_spec="all",
            cache=Stage2KVCache(),
        )


def test_concat_uses_native_then_masked_reference_bank_and_preserves_outside(tmp_path: Path) -> None:
    controller = _controller(tmp_path, mode="concat", strength=0.5)
    reference_k = torch.tensor([[[99.0, 99.0], [2.0, 0.0], [0.0, 2.0]]])
    reference_v = torch.tensor([[[99.0, 99.0], [10.0, 0.0], [0.0, 10.0]]])
    pre_q = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]
    )
    pre_k = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.5, 0.5], [0.0, -1.0]]]
    )
    values = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    post_q = pre_q + 10.0
    post_k = pre_k - 7.0

    controller.begin_step(0)
    controller.phase = "reference"
    controller.set_pre_rope(0, torch.zeros_like(reference_k), reference_k)
    controller.attend(
        layer_index=0,
        original=_attention,
        q=reference_k,
        k=reference_k,
        v=reference_v,
        heads=1,
        mask=None,
    )
    controller.phase = "hybrid"
    controller.set_pre_rope(0, pre_q, pre_k)
    controller._masked_attention[0] = _attention
    output = controller.attend(
        layer_index=0,
        original=_attention,
        q=post_q,
        k=post_k,
        v=values,
        heads=1,
        mask=None,
    )

    baseline = _attention(post_q, post_k, values, 1)
    inside = torch.tensor([1, 3])
    concat_k = torch.cat((pre_k, reference_k[:, 1:]), dim=1)
    concat_v = torch.cat((values, reference_v[:, 1:]), dim=1)
    concat_mask = torch.cat(
        (
            torch.zeros(inside.numel(), pre_k.shape[1]),
            torch.full((inside.numel(), 2), torch.log(torch.tensor(0.5))),
        ),
        dim=-1,
    )
    expected_inside = _attention(pre_q.index_select(1, inside), concat_k, concat_v, 1, concat_mask)
    assert torch.allclose(output[:, inside], expected_inside)
    assert torch.equal(output[:, [0, 2, 4, 5]], baseline[:, [0, 2, 4, 5]])


@pytest.mark.parametrize("mode", ["blend", "concat"])
def test_reference_zero_strength_is_exact_noop_for_all_guidance_modes(tmp_path: Path, mode: str) -> None:
    controller = _controller(tmp_path, mode=mode, strength=0.0)
    q = torch.randn(1, 6, 2)
    k = torch.randn(1, 6, 2)
    v = torch.randn(1, 6, 2)
    controller.begin_step(0)
    controller.phase = "hybrid"
    output = controller.attend(layer_index=0, original=_attention, q=q, k=k, v=v, heads=1, mask=None)
    assert torch.equal(output, _attention(q, k, v, 1))
    assert controller.cache.bytes_stored == 0


def test_concat_is_invariant_to_joint_reference_kv_permutation(tmp_path: Path) -> None:
    reference_k = torch.randn(1, 3, 4)
    reference_v = torch.randn(1, 3, 4)
    hybrid_q = torch.randn(1, 4, 4)
    hybrid_k = torch.randn(1, 4, 4)
    hybrid_v = torch.randn(1, 4, 4)
    original = _run_reference_then_hybrid(
        _controller(tmp_path, mode="concat", strength=1.0),
        reference_k=reference_k,
        reference_v=reference_v,
        hybrid_q=hybrid_q,
        hybrid_k=hybrid_k,
        hybrid_v=hybrid_v,
    )
    permutation = torch.tensor([0, 2, 1])
    permuted = _run_reference_then_hybrid(
        _controller(tmp_path, mode="concat", strength=1.0),
        reference_k=reference_k.index_select(1, permutation),
        reference_v=reference_v.index_select(1, permutation),
        hybrid_q=hybrid_q,
        hybrid_k=hybrid_k,
        hybrid_v=hybrid_v,
    )
    assert torch.allclose(original, permuted)


def test_post_rope_concat_uses_post_rope_bank_and_preserves_outside(tmp_path: Path) -> None:
    controller = _controller(tmp_path, mode="concat", strength=1.0, position_mode="post-rope")
    reference_k = torch.tensor([[[50.0, 50.0], [2.0, 0.0], [0.0, 2.0]]])
    reference_v = torch.tensor([[[99.0, 99.0], [10.0, 0.0], [0.0, 10.0]]])
    hybrid_q = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    hybrid_k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]])
    hybrid_v = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    output = _run_reference_then_hybrid(
        controller, reference_k=reference_k, reference_v=reference_v,
        hybrid_q=hybrid_q, hybrid_k=hybrid_k, hybrid_v=hybrid_v,
    )
    baseline = _attention(hybrid_q, hybrid_k, hybrid_v, 1)
    inside = torch.tensor([1, 3])
    expected = _attention(
        hybrid_q.index_select(1, inside),
        torch.cat((hybrid_k, reference_k[:, 1:]), dim=1),
        torch.cat((hybrid_v, reference_v[:, 1:]), dim=1), 1,
    )
    assert torch.allclose(output[:, inside], expected)
    assert torch.equal(output[:, [0, 2]], baseline[:, [0, 2]])
    assert controller.needs_pre_rope_features is False


def test_source_qk_reference_v_uses_anchor_routing_and_preserves_outside(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path, mode="source-qk-reference-v", strength=1.0, position_mode="post-rope"
    )
    anchor_q = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    anchor_k = torch.tensor([[[9.0, 9.0], [2.0, 0.0], [0.0, 2.0], [-9.0, -9.0]]])
    anchor_v = torch.zeros_like(anchor_k)
    reference_k = torch.randn(1, 3, 2)
    reference_v = torch.tensor([[[99.0, 99.0], [10.0, 0.0], [0.0, 10.0]]])
    hybrid_q = torch.randn(1, 4, 2)
    hybrid_k = torch.randn(1, 4, 2)
    hybrid_v = torch.randn(1, 4, 2)

    controller.begin_step(0)
    controller._unmasked_attention[0] = _attention
    controller.phase = "anchor"
    controller.attend(
        layer_index=0, original=_attention, q=anchor_q, k=anchor_k, v=anchor_v, heads=1, mask=None
    )
    controller.phase = "reference"
    controller.attend(
        layer_index=0, original=_attention, q=reference_k, k=reference_k, v=reference_v, heads=1, mask=None
    )
    controller.phase = "hybrid"
    output = controller.attend(
        layer_index=0, original=_attention, q=hybrid_q, k=hybrid_k, v=hybrid_v, heads=1, mask=None
    )

    inside = torch.tensor([1, 3])
    baseline = _attention(hybrid_q, hybrid_k, hybrid_v, 1)
    expected_inside = _attention(anchor_q[:, inside], anchor_k[:, 1:3], reference_v[:, 1:3], 1)
    assert torch.allclose(output[:, inside], expected_inside, atol=2e-3, rtol=2e-3)
    assert torch.equal(output[:, [0, 2]], baseline[:, [0, 2]])


def test_source_post_q_reference_pre_kv_uses_mixed_spaces_and_preserves_outside(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path, mode="source-q-reference-kv", strength=1.0, position_mode="q-post-k-pre"
    )
    anchor_post_q = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    anchor_post_k = torch.randn(1, 4, 2)
    reference_pre_k = torch.tensor([[[99.0, 99.0], [2.0, 0.0], [0.0, 2.0]]])
    reference_post_k = reference_pre_k + torch.tensor([[[0.0, 0.0], [-4.0, 4.0], [4.0, -4.0]]])
    reference_v = torch.tensor([[[99.0, 99.0], [10.0, 0.0], [0.0, 10.0]]])
    hybrid_q = torch.randn(1, 4, 2)
    hybrid_k = torch.randn(1, 4, 2)
    hybrid_v = torch.randn(1, 4, 2)

    controller.begin_step(0)
    controller._unmasked_attention[0] = _attention
    controller.phase = "anchor"
    controller.attend(
        layer_index=0, original=_attention, q=anchor_post_q, k=anchor_post_k,
        v=torch.zeros_like(anchor_post_k), heads=1, mask=None
    )
    controller.phase = "reference"
    controller.set_pre_rope(0, torch.zeros_like(reference_pre_k), reference_pre_k)
    controller.attend(
        layer_index=0, original=_attention, q=reference_post_k, k=reference_post_k,
        v=reference_v, heads=1, mask=None
    )
    controller.phase = "hybrid"
    output = controller.attend(
        layer_index=0, original=_attention, q=hybrid_q, k=hybrid_k, v=hybrid_v, heads=1, mask=None
    )

    inside = torch.tensor([1, 3])
    baseline = _attention(hybrid_q, hybrid_k, hybrid_v, 1)
    expected_inside = _attention(anchor_post_q[:, inside], reference_pre_k[:, 1:], reference_v[:, 1:], 1)
    wrong_post_rope = _attention(anchor_post_q[:, inside], reference_post_k[:, 1:], reference_v[:, 1:], 1)
    assert torch.allclose(output[:, inside], expected_inside, atol=2e-3, rtol=2e-3)
    assert not torch.allclose(output[:, inside], wrong_post_rope, atol=2e-3, rtol=2e-3)
    assert torch.equal(output[:, [0, 2]], baseline[:, [0, 2]])
