from __future__ import annotations

import pytest
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape
from ltx_pipelines.stage1_phi_noise import (
    Stage1PhiNoiseConfig,
    build_stage1_spatial_frequency_mask,
    construct_stage1_spatial_phi_noise,
    construct_stage1_temporal_phi_noise,
    make_stage1_noiser,
)


def _tools(frames: int = 5, height: int = 3, width: int = 4, channels: int = 2) -> VideoLatentTools:
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape(batch=1, channels=channels, frames=frames, height=height, width=width),
        fps=16.0,
    )


def test_own_phase_gamma_one_is_identity() -> None:
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(1, 3, 7, 4, 5, generator=generator)
    source = torch.randn(1, 3, 7, 4, 5, generator=generator)
    output, diagnostics = construct_stage1_temporal_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(enabled=True, k=2, gamma=1.0, phase_source="own"),
    )
    assert torch.allclose(output, noise, atol=1e-5)
    assert max(diagnostics["relative_energy_error"]) < 1e-6


@pytest.mark.parametrize("energy_scope", ["global", "per-series"])
def test_temporal_phi_preserves_energy_and_dtype(energy_scope: str) -> None:
    generator = torch.Generator().manual_seed(2)
    noise = torch.randn(2, 3, 7, 4, 5, generator=generator, dtype=torch.float16)
    source = torch.randn(2, 3, 7, 4, 5, generator=generator, dtype=torch.float16)
    output, diagnostics = construct_stage1_temporal_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(enabled=True, k=1, gamma=4.0, energy_scope=energy_scope),
    )
    assert output.shape == noise.shape
    assert output.dtype == noise.dtype
    assert diagnostics["fft_dtype"] == "torch.float32"
    original_energy = noise.float().square().sum()
    output_energy = output.float().square().sum()
    assert abs(float(output_energy - original_energy)) / float(original_energy) < 5e-4


def test_low_temporal_bins_match_source_phase_and_high_bins_keep_noise_phase() -> None:
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(1, 2, 7, 3, 4, generator=generator)
    source = torch.randn(1, 2, 7, 3, 4, generator=generator)
    output, diagnostics = construct_stage1_temporal_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(enabled=True, k=1, gamma=4.0, phase_source="source"),
    )
    z = torch.fft.rfft(noise.float(), dim=2)
    v = torch.fft.rfft(source.float(), dim=2)
    y = torch.fft.rfft(output.float(), dim=2)
    unit = lambda x: x / x.abs().clamp_min(1e-12)
    valid = (v[:, :, :2].abs() > 1e-12) & (z[:, :, :2].abs() > 1e-12)
    assert (unit(y[:, :, :2])[valid] - unit(v[:, :, :2])[valid]).abs().max() < 1e-5
    assert (unit(y[:, :, 2:]) - unit(z[:, :, 2:])).abs().max() < 1e-5
    assert diagnostics["max_low_phase_unit_error"] < 1e-5
    assert diagnostics["max_high_phase_unit_error"] < 1e-5


def test_exclude_frame0_leaves_first_latent_frame_unchanged() -> None:
    generator = torch.Generator().manual_seed(4)
    noise = torch.randn(1, 2, 7, 3, 4, generator=generator)
    source = torch.randn(1, 2, 7, 3, 4, generator=generator)
    output, _ = construct_stage1_temporal_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(enabled=True, k=1, gamma=4.0, exclude_frame0=True),
    )
    assert torch.equal(output[:, :, 0], noise[:, :, 0])


def test_patchify_unpatchify_order_round_trip() -> None:
    tools = _tools(frames=3, height=2, width=4, channels=1)
    grid = torch.zeros(1, 1, 3, 2, 4)
    for t in range(3):
        for h in range(2):
            for w in range(4):
                grid[0, 0, t, h, w] = 100 * t + 10 * h + w
    tokens = tools.patchifier.patchify(grid)
    assert tokens[0, 0, 0] == 0
    assert tokens[0, 1, 0] == 1
    assert tokens[0, 4, 0] == 10
    assert tokens[0, 8, 0] == 100
    assert torch.equal(tools.patchifier.unpatchify(tokens, output_shape=tools.target_shape), grid)


def test_make_stage1_noiser_disabled_matches_gaussian_noiser() -> None:
    tools = _tools()
    state = tools.create_initial_state("cpu", torch.float32)
    seed = 11
    gaussian = GaussianNoiser(torch.Generator().manual_seed(seed))(state)
    noiser = make_stage1_noiser(
        generator=torch.Generator().manual_seed(seed),
        video_tools=tools,
        source_latent=None,
        config=Stage1PhiNoiseConfig(enabled=False),
    )
    disabled = noiser(state)
    assert torch.equal(disabled.latent, gaussian.latent)


def test_stage1_noiser_changes_only_target_prefix_before_masking() -> None:
    tools = _tools(frames=5, height=2, width=3, channels=2)
    target_count = tools.target_shape.token_count()
    state = tools.create_initial_state("cpu", torch.float32)
    suffix = torch.zeros(1, 4, tools.target_shape.channels)
    suffix_mask = torch.ones(1, 4, state.denoise_mask.shape[-1])
    state = LatentState(
        latent=torch.cat([state.latent, suffix], dim=1),
        denoise_mask=torch.cat([state.denoise_mask, suffix_mask], dim=1),
        positions=torch.cat([state.positions, state.positions[:, :, :4]], dim=2),
        clean_latent=torch.cat([state.clean_latent, suffix.clone()], dim=1),
        attention_mask=None,
    )
    seed = 12
    gaussian = GaussianNoiser(torch.Generator().manual_seed(seed))(state)
    source = torch.randn(*tools.target_shape.to_torch_shape(), generator=torch.Generator().manual_seed(13))
    noiser = make_stage1_noiser(
        generator=torch.Generator().manual_seed(seed),
        video_tools=tools,
        source_latent=source,
        config=Stage1PhiNoiseConfig(enabled=True, k=1, gamma=4.0, phase_source="source"),
    )
    phi = noiser(state)
    assert not torch.equal(phi.latent[:, :target_count], gaussian.latent[:, :target_count])
    assert torch.equal(phi.latent[:, target_count:], gaussian.latent[:, target_count:])


@pytest.mark.parametrize(("height", "width"), [(4, 5), (5, 4), (5, 5), (6, 8)])
def test_spatial_mask_is_complete_ring_and_conjugate_symmetric(height: int, width: int) -> None:
    config = Stage1PhiNoiseConfig(enabled=True, transform="spatial", area_fraction=0.10)
    mask = build_stage1_spatial_frequency_mask(height, width, config, device="cpu")
    assert mask.actual_count == int(mask.values.sum())
    assert 0 < mask.actual_count < height * width
    radius = torch.sqrt(
        (torch.fft.fftshift(torch.fft.fftfreq(height, dtype=torch.float64))[:, None] / 0.5).square()
        + (torch.fft.fftshift(torch.fft.fftfreq(width, dtype=torch.float64))[None, :] / 0.5).square()
    )
    assert torch.equal(mask.values, radius <= mask.boundary_radius)
    unshifted = torch.fft.ifftshift(mask.values)
    neg_y = (-torch.arange(height)) % height
    neg_x = (-torch.arange(width)) % width
    assert torch.equal(unshifted, unshifted.index_select(0, neg_y).index_select(1, neg_x))


def test_spatial_own_phase_gamma_one_is_identity() -> None:
    generator = torch.Generator().manual_seed(31)
    noise = torch.randn(1, 2, 3, 6, 8, generator=generator)
    source = torch.randn(1, 2, 3, 6, 8, generator=generator)
    output, diagnostics = construct_stage1_spatial_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(enabled=True, transform="spatial", phase_source="own", gamma=1.0, area_fraction=0.10),
    )
    assert torch.allclose(output, noise, atol=1e-5)
    assert max(diagnostics["relative_energy_error"]) < 1e-6
    assert diagnostics["actual_fraction"] > 0.0


@pytest.mark.parametrize("energy_scope", ["global", "per-series"])
def test_spatial_phi_preserves_energy_and_dtype(energy_scope: str) -> None:
    generator = torch.Generator().manual_seed(32)
    noise = torch.randn(2, 3, 4, 6, 8, generator=generator, dtype=torch.float16)
    source = torch.randn(2, 3, 4, 6, 8, generator=generator, dtype=torch.float16)
    output, diagnostics = construct_stage1_spatial_phi_noise(
        noise,
        source,
        Stage1PhiNoiseConfig(
            enabled=True,
            transform="spatial",
            gamma=5.0,
            energy_scope=energy_scope,
            area_fraction=0.10,
        ),
    )
    assert output.shape == noise.shape
    assert output.dtype == noise.dtype
    original_energy = noise.float().square().sum()
    output_energy = output.float().square().sum()
    assert abs(float(output_energy - original_energy)) / float(original_energy) < 5e-4
    assert diagnostics["fft_dtype"] == "torch.float32"


def test_spatial_low_band_matches_source_phase_and_high_band_keeps_noise_phase() -> None:
    generator = torch.Generator().manual_seed(33)
    noise = torch.randn(1, 2, 3, 6, 8, generator=generator)
    source = torch.randn(1, 2, 3, 6, 8, generator=generator)
    config = Stage1PhiNoiseConfig(enabled=True, transform="spatial", gamma=5.0, area_fraction=0.10)
    output, diagnostics = construct_stage1_spatial_phi_noise(noise, source, config)
    mask = build_stage1_spatial_frequency_mask(6, 8, config, device="cpu").values.view(1, 1, 1, 6, 8)
    z = torch.fft.fftshift(torch.fft.fft2(noise.float(), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    v = torch.fft.fftshift(torch.fft.fft2(source.float(), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    y = torch.fft.fftshift(torch.fft.fft2(output.float(), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    unit = lambda x: x / x.abs().clamp_min(1e-12)
    low_valid = (v.abs() > 1e-12) & (z.abs() > 1e-12) & mask
    high = ~mask
    assert (unit(y)[low_valid] - unit(v)[low_valid]).abs().max() < 1e-5
    assert (unit(y)[high.expand_as(y)] - unit(z)[high.expand_as(z)]).abs().max() < 1e-5
    assert diagnostics["max_low_phase_unit_error"] < 1e-5
    assert diagnostics["max_high_phase_unit_error"] < 1e-5
