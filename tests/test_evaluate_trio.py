import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location('evaluate_trio', Path(__file__).parents[1] / 'scripts/_dual_traj/evaluate_trio.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_margin_and_iou():
    assert ev.appearance_margin(torch.tensor([1., 0.]), torch.tensor([1., 0.]), torch.tensor([0., 1.])) == (1., 0., 1.)
    empty = torch.zeros(2, 2)
    full = torch.ones(2, 2)
    assert ev.silhouette_iou(empty, empty) is None
    assert ev.silhouette_iou(empty, full) == 0
    assert ev.silhouette_iou(full, full) == 1


def test_soft_flow_and_scaling():
    src = torch.zeros(2, 2, 2)
    out = src.clone()
    out[0, 0, 0] = 4
    mask = torch.tensor([[.5, 0.], [1., 1.]])
    error, coverage = ev.flow_error(src, out, mask, mask)
    assert error == pytest.approx(2 / 1.5)
    assert coverage == pytest.approx(1.5 / 4)
    assert ev.flow_error(src, src, mask, mask)[0] == 0
    assert ev.flow_error(src, out, torch.ones_like(mask), mask)[0] is None
    resized = ev.resize_flow(torch.ones(1, 2, 2, 4), 6, 8)
    assert torch.all(resized[:, 0] == 2)
    assert torch.all(resized[:, 1] == 3)


def test_static_mask_and_crop_alignment(tmp_path):
    arr = np.zeros((10, 20), dtype=np.uint8)
    arr[:, 5:15] = 255
    path = tmp_path / 'mask.png'
    Image.fromarray(arr).save(path)
    m = ev.masks(path, 3, 8, 8)
    assert torch.equal(m[0], m[2])
    image = np.repeat(arr[..., None], 3, axis=-1)
    frame = ev.aligned_image(image, 8, 8)
    assert torch.allclose(frame[0], m[0])
    assert ev.appearance_input(frame, torch.zeros(8, 8)) is None


def test_temporal_resolution_and_cache_validation():
    good = dict(width=16, height=16, fps=16., frames=81)
    metas = {k: good.copy() for k in ('source', 'image', 'video', 'dual')}
    metas['source']['width'] = 32
    ev.validate_alignment(metas)
    metas['dual']['frames'] = 80
    with pytest.raises(ValueError, match='Temporal mismatch'):
        ev.validate_alignment(metas)
    metas['dual'] = dict(good, fps=24.)
    with pytest.raises(ValueError, match='Temporal mismatch'):
        ev.validate_alignment(metas)
    metas['dual'] = dict(good, width=32)
    with pytest.raises(ValueError, match='resolution mismatch'):
        ev.validate_alignment(metas)
    manifest = dict(source_video_sha256='abc', prompt_or_mask='dress')
    assert ev.cache_matches(manifest, 'abc', 'dress')
    assert not ev.cache_matches(manifest, 'def', 'dress')
    assert not ev.cache_matches(manifest, 'abc', 'person')
