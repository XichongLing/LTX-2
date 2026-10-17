from __future__ import annotations

import math

import pytest
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape
from ltx_pipelines.stage2_noise import (
    Stage2NoiseBundle,
    Stage2NoiseConfig,
    Stage2TargetNoiser,
    build_spatial_frequency_mask,
    construct_stage2_video_noise,
    load_stage2_noise_bundle,
    make_stage2_noiser,
    save_stage2_noise_bundle,
)


@pytest.mark.parametrize("area_fraction", [0.0, 1.0, -0.1, 1.1])
def test_area_fraction_is_strictly_between_zero_and_one(area_fraction: float) -> None:
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        Stage2NoiseConfig(area_fraction=area_fraction)


@pytest.mark.parametrize(("height", "width"), [(8, 8), (7, 9), (8, 9), (9, 8)])
def test_area_mask_is_deterministic_complete_ring_and_conjugate_symmetric(height: int, width: int) -> None:
    config = Stage2NoiseConfig(mode="phi", area_fraction=0.05)
    first = build_spatial_frequency_mask(height, width, config, device="cpu")
    second = build_spatial_frequency_mask(height, width, config, device="cpu")
    assert torch.equal(first.values, second.values)
    assert first.actual_count == int(first.values.sum())
    assert first.actual_count < height * width
    radius = torch.sqrt(
        (torch.fft.fftshift(torch.fft.fftfreq(height, dtype=torch.float64))[:, None] / 0.5).square()
        + (torch.fft.fftshift(torch.fft.fftfreq(width, dtype=torch.float64))[None, :] / 0.5).square()
    )
    assert torch.all(first.values == (radius <= first.boundary_radius))
    unshifted = torch.fft.ifftshift(first.values)
    neg_y = (-torch.arange(height)) % height
    neg_x = (-torch.arange(width)) % width
    assert torch.equal(unshifted, unshifted.index_select(0, neg_y).index_select(1, neg_x))


@pytest.mark.parametrize("alpha", [3, 4])
def test_official_alpha_reports_actual_corrected_frequency_fraction(alpha: int) -> None:
    config = Stage2NoiseConfig(mode="phi", mask_mode="official_alpha", alpha=alpha)
    mask = build_spatial_frequency_mask(16, 20, config, device="cpu")
    assert mask.requested_count is None
    assert mask.actual_fraction == pytest.approx(mask.actual_count / (16 * 20))
    assert mask.boundary_radius > 0.0


@pytest.mark.parametrize("mode", ["phi", "raw_reference_coefficients", "matched_reference_coefficients"])
def test_manipulated_modes_preserve_float32_spectral_energy_and_dtype(mode: str) -> None:
    generator = torch.Generator().manual_seed(12)
    gaussian = torch.randn(2, 3, 4, 12, 14, generator=generator, dtype=torch.float16)
    reference = torch.randn(2, 3, 4, 12, 14, generator=generator, dtype=torch.float16)
    output, diagnostics = construct_stage2_video_noise(
        gaussian,
        reference,
        Stage2NoiseConfig(mode=mode, area_fraction=0.05, gamma=5.0),
    )
    assert output.shape == gaussian.shape
    assert output.dtype == gaussian.dtype
    assert diagnostics["fft_dtype"] == "torch.float32"
    assert max(diagnostics["relative_energy_error"]) < 2e-6
    assert max(diagnostics["relative_imaginary_residual"]) < 1e-4
    assert all(math.isfinite(math_value) for math_value in diagnostics["beta"])


def test_phi_low_band_matches_reference_unit_complex_direction() -> None:
    generator = torch.Generator().manual_seed(21)
    gaussian = torch.randn(1, 2, 3, 14, 18, generator=generator)
    reference = torch.randn(1, 2, 3, 14, 18, generator=generator)
    config = Stage2NoiseConfig(mode="phi", area_fraction=0.10, gamma=5.0)
    output, _ = construct_stage2_video_noise(gaussian, reference, config)
    mask = build_spatial_frequency_mask(14, 18, config, device="cpu").values.view(1, 1, 1, 14, 18)
    output_fft = torch.fft.fftshift(torch.fft.fft2(output, norm="ortho"), dim=(-2, -1))
    reference_fft = torch.fft.fftshift(torch.fft.fft2(reference, norm="ortho"), dim=(-2, -1))
    gaussian_fft = torch.fft.fftshift(torch.fft.fft2(gaussian, norm="ortho"), dim=(-2, -1))
    eps = 1e-6
    valid = mask & (reference_fft.abs() > eps) & (gaussian_fft.abs() > eps)
    output_direction = output_fft / (output_fft.abs() + eps)
    reference_direction = reference_fft / (reference_fft.abs() + eps)
    assert torch.max(torch.abs(output_direction[valid] - reference_direction[valid])) < 1e-4


def test_gaussian_factory_is_exact_existing_noiser() -> None:
    shape = VideoLatentShape(batch=1, channels=4, frames=2, height=3, width=5)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    reference = torch.zeros(shape.to_torch_shape())
    noiser = make_stage2_noiser(
        generator=torch.Generator().manual_seed(4),
        video_tools=tools,
        reference_latent=reference,
        config=Stage2NoiseConfig(),
    )

    assert isinstance(noiser, GaussianNoiser)


def test_gaussian_diagnostics_remain_bit_exact_and_preserve_generator_order() -> None:
    shape = VideoLatentShape(batch=1, channels=4, frames=2, height=3, width=5)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    target_count = shape.token_count()
    state = LatentState(
        latent=torch.zeros(1, target_count, shape.channels),
        denoise_mask=torch.ones(1, target_count, 1),
        positions=torch.zeros(1, 3, target_count),
        clean_latent=torch.zeros(1, target_count, shape.channels),
    )
    diagnostics: list[dict[str, object]] = []
    observed_generator = torch.Generator().manual_seed(19)
    expected_generator = torch.Generator().manual_seed(19)
    observed = make_stage2_noiser(
        generator=observed_generator,
        video_tools=tools,
        reference_latent=torch.zeros(shape.to_torch_shape()),
        config=Stage2NoiseConfig(diagnostics_sink=diagnostics),
    )(state, 0.7)
    expected = GaussianNoiser(expected_generator)(state, 0.7)
    assert torch.equal(observed.latent, expected.latent)
    assert torch.equal(torch.randn(11, generator=observed_generator), torch.randn(11, generator=expected_generator))
    assert diagnostics[0]["gaussian_checksum"] == diagnostics[0]["output_checksum"]
    assert diagnostics[0]["relative_energy_error"] == [0.0]


def test_phi_noiser_preserves_suffix_draw_and_generator_order() -> None:
    shape = VideoLatentShape(batch=1, channels=4, frames=2, height=3, width=5)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    target_count = shape.token_count()
    total_count = target_count + 7
    state = LatentState(
        latent=torch.zeros(1, total_count, shape.channels),
        denoise_mask=torch.ones(1, total_count, 1),
        positions=torch.zeros(1, 3, total_count),
        clean_latent=torch.zeros(1, total_count, shape.channels),
    )
    custom_generator = torch.Generator().manual_seed(44)
    control_generator = torch.Generator().manual_seed(44)
    diagnostics: list[dict[str, object]] = []
    noiser = Stage2TargetNoiser(
        generator=custom_generator,
        video_tools=tools,
        reference_latent=torch.randn(shape.to_torch_shape(), generator=torch.Generator().manual_seed(9)),
        config=Stage2NoiseConfig(mode="phi", diagnostics_sink=diagnostics),
    )
    result = noiser(state)
    ordinary_draw = torch.randn(state.latent.shape, generator=control_generator)
    assert not torch.equal(result.latent[:, :target_count], ordinary_draw[:, :target_count])
    assert torch.equal(result.latent[:, target_count:], ordinary_draw[:, target_count:])
    assert torch.equal(torch.randn(13, generator=custom_generator), torch.randn(13, generator=control_generator))
    assert len(diagnostics) == 1


def test_noise_bundle_capture_preserves_legacy_combined_draw_order_and_replays(tmp_path) -> None:
    shape = VideoLatentShape(batch=1, channels=4, frames=2, height=3, width=5)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    target_count = shape.token_count()
    video_state = LatentState(
        latent=torch.zeros(1, target_count + 7, shape.channels),
        denoise_mask=torch.ones(1, target_count + 7, 1),
        positions=torch.zeros(1, 3, target_count + 7),
        clean_latent=torch.zeros(1, target_count + 7, shape.channels),
    )
    audio_state = LatentState(
        latent=torch.zeros(1, 9, shape.channels),
        denoise_mask=torch.ones(1, 9, 1),
        positions=torch.zeros(1, 3, 9),
        clean_latent=torch.zeros(1, 9, shape.channels),
    )
    bundle = Stage2NoiseBundle(provenance={"layout": "legacy"})
    observed_generator = torch.Generator().manual_seed(81)
    expected_generator = torch.Generator().manual_seed(81)
    noiser = make_stage2_noiser(
        generator=observed_generator,
        video_tools=tools,
        reference_latent=torch.zeros(shape.to_torch_shape()),
        config=Stage2NoiseConfig(noise_bundle=bundle),
    )
    observed_video = noiser(video_state)
    observed_audio = noiser(audio_state)
    expected_video = GaussianNoiser(expected_generator)(video_state)
    expected_audio = GaussianNoiser(expected_generator)(audio_state)
    assert torch.equal(observed_video.latent, expected_video.latent)
    assert torch.equal(observed_audio.latent, expected_audio.latent)
    assert torch.equal(torch.randn(5, generator=observed_generator), torch.randn(5, generator=expected_generator))
    assert torch.equal(bundle.target_video, expected_video.latent[:, :target_count])
    assert torch.equal(bundle.source_video, expected_video.latent[:, target_count:])
    assert torch.equal(bundle.audio, expected_audio.latent)

    path = tmp_path / "noise.safetensors"
    save_stage2_noise_bundle(path, bundle)
    replay = load_stage2_noise_bundle(path)
    replay_noiser = make_stage2_noiser(
        generator=torch.Generator().manual_seed(999),
        video_tools=tools,
        reference_latent=torch.zeros(shape.to_torch_shape()),
        config=Stage2NoiseConfig(noise_bundle=replay),
    )
    assert torch.equal(replay_noiser(video_state).latent, expected_video.latent)
    assert torch.equal(replay_noiser(audio_state).latent, expected_audio.latent)


def test_reference_shape_must_exactly_match_noise() -> None:
    gaussian = torch.randn(1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="exactly match"):
        construct_stage2_video_noise(
            gaussian,
            torch.randn(1, 2, 2, 8, 8),
            Stage2NoiseConfig(mode="phi"),
        )
