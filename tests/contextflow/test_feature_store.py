import pytest
import torch

from ltx_pipelines.inversion.attention.feature_store import FeatureStore


def test_put_get_roundtrip() -> None:
    store = FeatureStore()
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    store.put(step_index=0, layer_index=1, k=k, v=v, representation_stage="post_rope")
    stored = store.get(step_index=0, layer_index=1)
    assert torch.equal(stored.k, k)
    assert torch.equal(stored.v, v)


def test_wrong_step_raises() -> None:
    store = FeatureStore()
    store.put(
        step_index=0,
        layer_index=1,
        k=torch.randn(1, 4, 8),
        v=torch.randn(1, 4, 8),
        representation_stage="post_rope",
    )
    with pytest.raises(KeyError):
        store.get(step_index=2, layer_index=1)


def test_wrong_layer_raises() -> None:
    store = FeatureStore()
    store.put(
        step_index=0,
        layer_index=1,
        k=torch.randn(1, 4, 8),
        v=torch.randn(1, 4, 8),
        representation_stage="post_rope",
    )
    with pytest.raises(KeyError):
        store.get(step_index=0, layer_index=3)


def test_clear_step_removes_only_current_step() -> None:
    store = FeatureStore()
    store.put(step_index=0, layer_index=0, k=torch.randn(1, 4, 8), v=torch.randn(1, 4, 8), representation_stage="post_rope")
    store.put(step_index=1, layer_index=0, k=torch.randn(1, 4, 8), v=torch.randn(1, 4, 8), representation_stage="post_rope")
    store.clear_step(0)
    with pytest.raises(KeyError):
        store.get(step_index=0, layer_index=0)
    assert store.get(step_index=1, layer_index=0).step_index == 1


def test_clear_all_removes_everything() -> None:
    store = FeatureStore()
    store.put(step_index=0, layer_index=0, k=torch.randn(1, 4, 8), v=torch.randn(1, 4, 8), representation_stage="post_rope")
    store.put(step_index=1, layer_index=0, k=torch.randn(1, 4, 8), v=torch.randn(1, 4, 8), representation_stage="post_rope")
    store.clear_all()
    assert store.is_empty()
