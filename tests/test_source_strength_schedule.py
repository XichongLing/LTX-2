import math

import pytest
import torch

from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor
from ltx_core.types import LatentState
from ltx_pipelines.utils.conditioning_schedules import (
    ConstantFirstFrameAttentionSchedule,
    ConstantSourceStrengthSchedule,
    SigmaRampFirstFrameAttentionSchedule,
    SigmaRampSourceStrengthSchedule,
    SourceStrengthRouting,
    ValuesFirstFrameAttentionSchedule,
    ValuesSourceStrengthSchedule,
)
from ltx_pipelines.utils.denoisers import SourceStrengthScheduledDenoiser
from ltx_pipelines.utils.types import DenoisedLatentResult


class CapturingInner:
    def __init__(self) -> None:
        self.states = []

    def __call__(self, _transformer, video_state, audio_state, sigmas, step_index):
        self.states.append(video_state)
        return DenoisedLatentResult(video_state.latent), None


def _state(attention_mask=None) -> LatentState:
    latent = torch.ones((1, 7, 2), dtype=torch.float32)
    return LatentState(
        latent=latent,
        denoise_mask=torch.ones((1, 7, 1), dtype=torch.float32),
        positions=torch.zeros((1, 3, 7), dtype=torch.float32),
        clean_latent=latent.clone(),
        attention_mask=attention_mask,
    )


def _run(schedule, *, routing=SourceStrengthRouting.SYMMETRIC, attention_mask=None, num_evaluations=1):
    inner = CapturingInner()
    rows = []
    denoiser = SourceStrengthScheduledDenoiser(
        inner,
        schedule=schedule,
        target_token_count=4,
        source_token_counts=(2,),
        routing=routing,
        num_evaluations=num_evaluations,
        sink=rows.append,
    )
    denoiser(object(), _state(attention_mask), None, torch.tensor([1.0, 0.0]), 0)
    return inner.states[-1], rows




def _run_with_first_frame(
    first_frame_schedule,
    *,
    source_schedule=ConstantSourceStrengthSchedule(1.0),
    attention_mask=None,
    num_evaluations=1,
):
    inner = CapturingInner()
    source_rows = []
    frame_rows = []
    denoiser = SourceStrengthScheduledDenoiser(
        inner,
        schedule=source_schedule,
        target_token_count=4,
        source_token_counts=(2,),
        routing=SourceStrengthRouting.SYMMETRIC,
        num_evaluations=num_evaluations,
        sink=source_rows.append,
        first_frame_schedule=first_frame_schedule,
        first_frame_token_count=2,
        first_frame_sink=frame_rows.append,
    )
    denoiser(object(), _state(attention_mask), None, torch.tensor([1.0, 0.0]), 0)
    return inner.states[-1], source_rows, frame_rows


def test_constant_one_preserves_original_state_and_mask_path() -> None:
    mask = torch.ones((1, 7, 7))
    state, rows = _run(ConstantSourceStrengthSchedule(1.0), attention_mask=mask)

    assert state.attention_mask is mask
    assert rows[0].g_source == 1.0
    assert rows[0].source_token_ranges == ((5, 7),)
    assert rows[0].target_token_count == 4


def test_hard_disable_blocks_only_target_source_interactions() -> None:
    state, rows = _run(ConstantSourceStrengthSchedule(0.0))
    mask = state.attention_mask

    assert mask is not None
    assert torch.equal(mask[:, :4, 5:7], torch.zeros((1, 4, 2)))
    assert torch.equal(mask[:, 5:7, :4], torch.zeros((1, 2, 4)))
    assert torch.equal(mask[:, :4, :4], torch.ones((1, 4, 4)))
    assert torch.equal(mask[:, 4:5, 5:7], torch.ones((1, 1, 2)))
    assert torch.equal(mask[:, 5:7, 5:7], torch.ones((1, 2, 2)))
    assert rows[0].composed_existing_mask is False


def test_intermediate_strength_composes_existing_spatial_mask() -> None:
    existing = torch.ones((1, 7, 7))
    existing[:, :4, 5:7] = 0.5
    existing[:, 5:7, :4] = 0.25

    state, rows = _run(ConstantSourceStrengthSchedule(0.5), attention_mask=existing)

    assert torch.equal(state.attention_mask[:, :4, 5:7], torch.full((1, 4, 2), 0.25))
    assert torch.equal(state.attention_mask[:, 5:7, :4], torch.full((1, 2, 4), 0.125))
    assert rows[0].composed_existing_mask is True


def test_target_queries_only_routing_leaves_source_queries_unchanged() -> None:
    state, _rows = _run(
        ConstantSourceStrengthSchedule(0.25),
        routing=SourceStrengthRouting.TARGET_QUERIES_ONLY,
    )

    assert torch.equal(state.attention_mask[:, :4, 5:7], torch.full((1, 4, 2), 0.25))
    assert torch.equal(state.attention_mask[:, 5:7, :4], torch.ones((1, 2, 4)))


def test_log_space_conversion_matches_attention_weight_semantics() -> None:
    preprocessor = TransformerArgsPreprocessor.__new__(TransformerArgsPreprocessor)
    weights = torch.tensor([[[1.0, 0.5, 0.0]]])

    bias = preprocessor._prepare_self_attention_mask(weights, torch.float32)

    assert bias.shape == (1, 1, 1, 3)
    assert bias[0, 0, 0, 0] == 0.0
    assert torch.isclose(bias[0, 0, 0, 1], torch.tensor(math.log(0.5)))
    assert torch.isfinite(bias[0, 0, 0, 2])
    assert bias[0, 0, 0, 2] < -1e30


def test_values_schedule_rejects_mismatched_evaluation_count() -> None:
    with pytest.raises(ValueError, match="must match"):
        _run(ValuesSourceStrengthSchedule((1.0, 0.5)), num_evaluations=3)


def test_sigma_ramp_uses_sigma_not_step_count() -> None:
    schedule = SigmaRampSourceStrengthSchedule(
        early_strength=1.0,
        late_strength=0.3,
        fade_start_sigma=0.8,
        fade_end_sigma=0.4,
    )

    assert schedule.strength(sigma=0.9, progress=0.0, evaluation_index=0, num_evaluations=8) == 1.0
    assert schedule.strength(sigma=0.3, progress=1.0, evaluation_index=7, num_evaluations=8) == 0.3
    assert schedule.strength(sigma=0.6, progress=0.5, evaluation_index=3, num_evaluations=8) == pytest.approx(0.65)


def test_schedule_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ConstantSourceStrengthSchedule(float("nan"))
    assert ConstantSourceStrengthSchedule(4.0).value == 4.0
    with pytest.raises(ValueError):
        ConstantSourceStrengthSchedule(-0.1)
    with pytest.raises(ValueError):
        ValuesSourceStrengthSchedule(())
    with pytest.raises(ValueError):
        SigmaRampSourceStrengthSchedule(fade_start_sigma=0.1, fade_end_sigma=0.2)



def test_first_frame_multiplier_boosts_only_later_target_queries_to_frame0_keys() -> None:
    state, _source_rows, frame_rows = _run_with_first_frame(ConstantFirstFrameAttentionSchedule(3.0))
    mask = state.attention_mask

    assert mask is not None
    assert torch.equal(mask[:, 2:4, 0:2], torch.full((1, 2, 2), 3.0))
    assert torch.equal(mask[:, 0:2, 0:2], torch.ones((1, 2, 2)))
    assert torch.equal(mask[:, 0:2, 2:4], torch.ones((1, 2, 2)))
    assert torch.equal(mask[:, 4:5, 0:2], torch.ones((1, 1, 2)))
    assert torch.equal(mask[:, 5:7, 0:2], torch.ones((1, 2, 2)))
    assert frame_rows[0].h_first_frame == 3.0
    assert frame_rows[0].frame0_token_range == (0, 2)
    assert frame_rows[0].later_target_token_range == (2, 4)


def test_first_frame_multiplier_composes_existing_mask_and_source_schedule_once() -> None:
    existing = torch.ones((1, 7, 7))
    existing[:, 2:4, 0:2] = 0.5
    existing[:, :4, 5:7] = 0.25

    state, source_rows, frame_rows = _run_with_first_frame(
        ConstantFirstFrameAttentionSchedule(2.0),
        source_schedule=ConstantSourceStrengthSchedule(0.5),
        attention_mask=existing,
    )

    assert torch.equal(state.attention_mask[:, 2:4, 0:2], torch.ones((1, 2, 2)))
    assert torch.equal(state.attention_mask[:, :4, 5:7], torch.full((1, 4, 2), 0.125))
    assert source_rows[0].composed_existing_mask is True
    assert frame_rows[0].composed_existing_mask is True


def test_first_frame_values_schedule_rejects_mismatched_evaluation_count() -> None:
    with pytest.raises(ValueError, match="must match"):
        _run_with_first_frame(ValuesFirstFrameAttentionSchedule((1.0, 2.0)), num_evaluations=3)


def test_first_frame_sigma_ramp_allows_amplification() -> None:
    schedule = SigmaRampFirstFrameAttentionSchedule(
        early_multiplier=1.0,
        late_multiplier=4.0,
        fade_start_sigma=0.8,
        fade_end_sigma=0.4,
    )

    assert schedule.multiplier(sigma=0.9, progress=0.0, evaluation_index=0, num_evaluations=8) == 1.0
    assert schedule.multiplier(sigma=0.3, progress=1.0, evaluation_index=7, num_evaluations=8) == 4.0
    assert schedule.multiplier(sigma=0.6, progress=0.5, evaluation_index=3, num_evaluations=8) == pytest.approx(2.5)


def test_first_frame_schedule_validation_rejects_invalid_multipliers() -> None:
    with pytest.raises(ValueError):
        ConstantFirstFrameAttentionSchedule(float("inf"))
    with pytest.raises(ValueError):
        ConstantFirstFrameAttentionSchedule(17.0)
    with pytest.raises(ValueError):
        ValuesFirstFrameAttentionSchedule(())
    with pytest.raises(ValueError):
        SigmaRampFirstFrameAttentionSchedule(fade_start_sigma=0.1, fade_end_sigma=0.2)


def test_first_frame_schedule_allows_single_frame_target() -> None:
    denoiser = SourceStrengthScheduledDenoiser(
        CapturingInner(),
        schedule=ConstantSourceStrengthSchedule(1.0),
        target_token_count=4,
        source_token_counts=(2,),
        first_frame_schedule=ConstantFirstFrameAttentionSchedule(1.0),
        first_frame_token_count=4,
    )
    assert denoiser._frame0_ranges() == ((0, 4), (4, 4))
