import torch

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_pipelines.inversion.schedule import LayerStepSchedule
from ltx_pipelines.inversion.schedule import CONTEXTFLOW_SIGMA_MIN, contextflow_sigmas


def test_schedule_boundaries() -> None:
    schedule = LayerStepSchedule(active_layers=frozenset({4}), start_fraction=0.25, end_fraction=0.5)
    assert schedule.is_active(step_index=1, num_steps=5, layer_index=4)
    assert schedule.is_active(step_index=2, num_steps=5, layer_index=4)
    assert not schedule.is_active(step_index=0, num_steps=5, layer_index=4)
    assert not schedule.is_active(step_index=4, num_steps=5, layer_index=4)


def test_schedule_inactive_layer() -> None:
    schedule = LayerStepSchedule(active_layers=frozenset({4}), start_fraction=0.0, end_fraction=1.0)
    assert not schedule.is_active(step_index=0, num_steps=4, layer_index=3)


def test_schedule_layer_range() -> None:
    schedule = LayerStepSchedule(layer_range=(2, 4), start_fraction=0.0, end_fraction=1.0)
    assert schedule.is_active(step_index=0, num_steps=2, layer_index=2)
    assert schedule.is_active(step_index=0, num_steps=2, layer_index=4)
    assert not schedule.is_active(step_index=0, num_steps=2, layer_index=5)


def test_contextflow_large_latent_schedule_is_finite_and_strictly_decreasing() -> None:
    latent = torch.empty(1, 128, 15, 44, 80, device="meta")

    sigmas = contextflow_sigmas(LTX2Scheduler(), steps=10, latent=latent)

    assert sigmas.shape == (11,)
    assert torch.isfinite(sigmas).all()
    assert torch.all(sigmas > 0)
    assert torch.all(sigmas[1:] < sigmas[:-1])
    assert torch.isclose(sigmas[-1], torch.tensor(CONTEXTFLOW_SIGMA_MIN, dtype=sigmas.dtype))
