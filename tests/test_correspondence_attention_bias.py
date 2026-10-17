import math

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning.types.correspondence_attention_bias_wrapper import (
    apply_target_to_reference_correspondence_bias,
)
from ltx_core.types import LatentState, VideoLatentShape
from ltx_pipelines.correspondence_mask import (
    CorrespondenceMaskBox,
    build_box_mask,
    build_later_frames_mask,
    build_masked_denoise_mask,
    downsample_disocclusion_mask_to_target_tokens,
)


def test_later_frames_mask_preserves_causal_frame_zero() -> None:
    mask = build_later_frames_mask(num_frames=17, height=2, width=3)
    target_shape = VideoLatentShape(batch=1, channels=128, frames=3, height=1, width=1)

    latent_mask = downsample_disocclusion_mask_to_target_tokens(mask, target_shape)

    assert torch.equal(latent_mask, torch.tensor([[0.0, 1.0, 1.0]]))


def test_temporal_max_pool_preserves_brief_disocclusion() -> None:
    mask = torch.zeros((1, 1, 17, 1, 1))
    mask[:, :, 5] = 1
    target_shape = VideoLatentShape(batch=1, channels=128, frames=3, height=1, width=1)

    latent_mask = downsample_disocclusion_mask_to_target_tokens(mask, target_shape)

    assert torch.equal(latent_mask, torch.tensor([[0.0, 1.0, 0.0]]))


def test_box_mask_uses_normalized_coordinates_and_inclusive_frames() -> None:
    box = CorrespondenceMaskBox(start_frame=1, end_frame=2, x0=0.25, y0=0.25, x1=0.75, y1=0.75)

    mask = build_box_mask(boxes=[box], num_frames=4, height=4, width=4)

    assert mask[:, :, 0].sum() == 0
    assert mask[:, :, 1:3, 1:3, 1:3].sum() == 8
    assert mask[:, :, 3].sum() == 0


def test_correspondence_bias_is_directional_and_mask_gated() -> None:
    target_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=2, width=2)
    reference_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=1, width=1)
    target_mask = torch.tensor([[0.0, 1.0, 0.5, 0.0]])
    reference_start = 6

    weights = apply_target_to_reference_correspondence_bias(
        None,
        total_tokens=7,
        num_target_tokens=4,
        reference_start=reference_start,
        target_mask=target_mask,
        target_shape=target_shape,
        reference_shape=reference_shape,
        logit_bias=math.log(4.0),
        radius=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.allclose(weights[0, :4, reference_start], torch.tensor([1.0, 4.0, 2.0, 1.0]))
    assert torch.equal(weights[0, reference_start, :4], torch.ones(4))
    unchanged = weights.clone()
    unchanged[0, :4, reference_start] = 1
    assert torch.equal(unchanged, torch.ones_like(unchanged))


def test_correspondence_bias_multiplies_existing_weights() -> None:
    target_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=1, width=1)
    reference_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=1, width=1)
    existing = torch.full((1, 2, 2), 0.5)

    weights = apply_target_to_reference_correspondence_bias(
        existing,
        total_tokens=2,
        num_target_tokens=1,
        reference_start=1,
        target_mask=torch.ones((1, 1)),
        target_shape=target_shape,
        reference_shape=reference_shape,
        logit_bias=math.log(2.0),
        radius=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.isclose(weights[0, 0, 1], torch.tensor(1.0))
    assert torch.isclose(weights[0, 1, 0], torch.tensor(0.5))


def test_radius_one_biases_valid_reference_neighbors() -> None:
    target_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=2, width=2)
    reference_shape = VideoLatentShape(batch=1, channels=128, frames=1, height=2, width=2)

    weights = apply_target_to_reference_correspondence_bias(
        None,
        total_tokens=8,
        num_target_tokens=4,
        reference_start=4,
        target_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        target_shape=target_shape,
        reference_shape=reference_shape,
        logit_bias=math.log(2.0),
        radius=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.equal(weights[0, 0, 4:8], torch.full((4,), 2.0))
    assert torch.equal(weights[0, 1, 4:8], torch.ones(4))


def test_masked_denoise_mask_only_reduces_masked_target_tokens() -> None:
    mask = torch.zeros((1, 1, 17, 2, 2))
    mask[:, :, 1:, 0, 0] = 1
    target_shape = VideoLatentShape(batch=1, channels=128, frames=3, height=2, width=2)

    target_denoise_mask = build_masked_denoise_mask(mask, target_shape, masked_strength=0.25)

    expected = torch.ones((1, 12))
    expected[:, 4] = 0.25
    expected[:, 8] = 0.25
    assert torch.equal(target_denoise_mask, expected)


def test_denoise_mask_does_not_change_later_denoise_mask() -> None:
    latent = torch.ones((1, 4, 2))
    denoise_mask = torch.tensor([[[0.0], [0.5], [1.0], [1.0]]])
    state = LatentState(
        latent=latent,
        denoise_mask=denoise_mask,
        positions=torch.zeros((1, 3, 4)),
        clean_latent=latent.clone(),
    )
    expected_generator = torch.Generator(device="cpu").manual_seed(7)
    expected_noise = torch.randn(latent.shape, generator=expected_generator)
    actual_generator = torch.Generator(device="cpu").manual_seed(7)

    result = GaussianNoiser(actual_generator)(state, noise_scale=1.0)

    expected = expected_noise.clone()
    expected[:, 0] = latent[:, 0]
    expected[:, 1] = 0.5 * expected_noise[:, 1] + 0.5 * latent[:, 1]
    assert torch.allclose(result.latent, expected)
    assert torch.equal(result.denoise_mask, denoise_mask)


def test_masked_denoise_strength_one_is_identity_mask() -> None:
    mask = torch.ones((1, 1, 17, 2, 2))
    target_shape = VideoLatentShape(batch=1, channels=128, frames=3, height=2, width=2)

    target_denoise_mask = build_masked_denoise_mask(mask, target_shape, masked_strength=1.0)

    assert torch.equal(target_denoise_mask, torch.ones_like(target_denoise_mask))
