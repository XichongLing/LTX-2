from pathlib import Path

import pytest
import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.stage2_kv import (
    Stage2KVAttentionController,
    Stage2KVCache,
    parse_stage2_kv_layers,
)
from ltx_pipelines.stage2_routing import downsample_semantic_mask_to_hard_target_tokens


def _mean_attention(
    q: torch.Tensor,
    _k: torch.Tensor,
    v: torch.Tensor,
    _heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    del mask
    return v.mean(dim=1, keepdim=True).expand(-1, q.shape[1], -1)


def _controller(
    tmp_path: Path, *, mode: str, strength: float = 1.0, backend: str = "cpu"
) -> Stage2KVAttentionController:
    cache = Stage2KVCache(backend=backend, cache_dir=tmp_path if backend == "disk" else None)
    return Stage2KVAttentionController(
        target_token_count=4,
        hard_mask=torch.tensor([[False, True, False, True]]),
        mode=mode,
        strength=strength,
        layer_spec="all",
        cache=cache,
    )


@pytest.mark.parametrize("backend", ["cpu", "disk"])
def test_kv_cache_roundtrip_and_cleanup(tmp_path: Path, backend: str) -> None:
    cache = Stage2KVCache(backend=backend, cache_dir=tmp_path if backend == "disk" else None)
    k = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    v = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    cache.begin_step(2)
    cache.put(4, k, v)
    loaded_k, loaded_v = cache.consume(4, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert torch.equal(loaded_k, k)
    assert torch.equal(loaded_v, v)
    assert cache.bytes_stored == (k.numel() + v.numel()) * k.element_size()
    cache.clear()
    assert not (tmp_path / "step_002").exists()


@pytest.mark.parametrize("mode", ["partition", "append"])
def test_guidance_changes_only_dress_queries(tmp_path: Path, mode: str) -> None:
    controller = _controller(tmp_path, mode=mode)
    q = torch.zeros(1, 4, 2)
    image_k = torch.zeros(1, 4, 2)
    image_v = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    anchor_k = torch.ones(1, 6, 2)
    anchor_v = torch.full((1, 6, 2), 20.0)

    controller.cache.begin_step(0)
    controller.phase = "capture"
    controller.attend(
        layer_index=0,
        original=_mean_attention,
        q=torch.zeros(1, 6, 2),
        k=anchor_k,
        v=anchor_v,
        heads=1,
        mask=None,
    )
    controller.phase = "inject"
    output = controller.attend(
        layer_index=0,
        original=_mean_attention,
        q=q,
        k=image_k,
        v=image_v,
        heads=1,
        mask=None,
    )
    baseline = _mean_attention(q, image_k, image_v, 1)
    assert torch.equal(output[:, [0, 2]], baseline[:, [0, 2]])
    assert not torch.equal(output[:, [1, 3]], baseline[:, [1, 3]])


def test_zero_strength_is_exact_noop_and_does_not_cache(tmp_path: Path) -> None:
    controller = _controller(tmp_path, mode="append", strength=0.0)
    controller.cache.begin_step(0)
    controller.phase = "capture"
    q = k = v = torch.randn(1, 4, 2)
    expected = _mean_attention(q, k, v, 1)
    actual = controller.attend(layer_index=0, original=_mean_attention, q=q, k=k, v=v, heads=1, mask=None)
    assert torch.equal(actual, expected)
    assert controller.cache.bytes_stored == 0


def test_hard_mask_uses_causal_temporal_and_spatial_max_pooling() -> None:
    shape = VideoLatentShape(batch=1, channels=2, frames=2, height=1, width=1)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    mask = torch.tensor(
        [[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.6], [0.0, 0.0]]]]]
    )
    hard = downsample_semantic_mask_to_hard_target_tokens(mask, tools)
    assert torch.equal(hard, torch.tensor([[False, True]]))


def test_layer_spec_parser() -> None:
    assert parse_stage2_kv_layers("0,2-4", 6) == {0, 2, 3, 4}
    assert parse_stage2_kv_layers("all", 3) == {0, 1, 2}
    with pytest.raises(ValueError, match="out of range"):
        parse_stage2_kv_layers("3", 3)
