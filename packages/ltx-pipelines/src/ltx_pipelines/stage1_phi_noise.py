"""Stage-1 temporal Phi-noise for target-video initialization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import load_file

from ltx_core.components.noisers import GaussianNoiser, Noiser
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState


class Stage1PhiNoiseTransform(str, Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class Stage1PhiNoiseEnergyScope(str, Enum):
    GLOBAL = "global"
    PER_SERIES = "per-series"


class Stage1PhiNoiseMaskMode(str, Enum):
    AREA_FRACTION = "area_fraction"
    OFFICIAL_ALPHA = "official_alpha"


class Stage1PhiNoisePhaseSource(str, Enum):
    SOURCE = "source"
    OWN = "own"


@dataclass(frozen=True)
class SpatialFrequencyMask:
    values: torch.Tensor
    requested_count: int | None
    actual_count: int
    actual_fraction: float
    boundary_radius: float


@dataclass(frozen=True)
class Stage1PhiNoiseConfig:
    enabled: bool = False
    transform: Stage1PhiNoiseTransform | str = Stage1PhiNoiseTransform.TEMPORAL
    k: int = 1
    gamma: float = 30.0
    energy_scope: Stage1PhiNoiseEnergyScope | str = Stage1PhiNoiseEnergyScope.GLOBAL
    exclude_frame0: bool = False
    phase_source: Stage1PhiNoisePhaseSource | str = Stage1PhiNoisePhaseSource.SOURCE
    mask_mode: Stage1PhiNoiseMaskMode | str = Stage1PhiNoiseMaskMode.AREA_FRACTION
    area_fraction: float = 0.05
    alpha: int = 3
    source_latent_path: str | None = None
    diagnostics_sink: list[dict[str, object]] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", Stage1PhiNoiseTransform(self.transform))
        object.__setattr__(self, "energy_scope", Stage1PhiNoiseEnergyScope(self.energy_scope))
        object.__setattr__(self, "phase_source", Stage1PhiNoisePhaseSource(self.phase_source))
        object.__setattr__(self, "mask_mode", Stage1PhiNoiseMaskMode(self.mask_mode))
        if self.k < 0:
            raise ValueError(f"Stage-1 Phi-noise k must be non-negative, got {self.k}")
        if not math.isfinite(self.gamma) or self.gamma < 1.0:
            raise ValueError(f"Stage-1 Phi-noise gamma must be finite and >= 1, got {self.gamma}")
        if not 0.0 < self.area_fraction < 1.0:
            raise ValueError(f"Stage-1 Phi-noise area_fraction must be in (0,1), got {self.area_fraction}")
        if self.alpha <= 0:
            raise ValueError(f"Stage-1 Phi-noise alpha must be positive, got {self.alpha}")

    def manifest_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "transform": self.transform.value,
            "k": self.k,
            "gamma": self.gamma,
            "energy_scope": self.energy_scope.value,
            "exclude_frame0": self.exclude_frame0,
            "phase_source": self.phase_source.value,
            "mask_mode": self.mask_mode.value,
            "area_fraction": self.area_fraction,
            "alpha": self.alpha,
            "source_latent_path": self.source_latent_path,
        }


def tensor_checksum(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def load_stage1_phi_source_latent(path: str | Path) -> torch.Tensor:
    latent_path = Path(path).expanduser().resolve()
    if latent_path.suffix == ".safetensors":
        tensors = load_file(latent_path, device="cpu")
        if "latent" in tensors:
            return tensors["latent"]
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        raise ValueError(f"Phi source safetensors must contain key 'latent' or exactly one tensor: {latent_path}")
    loaded = torch.load(latent_path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        return loaded
    if isinstance(loaded, dict):
        for key in ("latent", "video", "video_latent"):
            value = loaded.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"unsupported Phi source latent file contents: {latent_path}")


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "min": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "max": float(flat.max().item()),
    }


def _per_channel_mean_std(value: torch.Tensor) -> dict[str, list[float]]:
    return {
        "mean": [float(x) for x in value.float().mean(dim=(0, 2, 3, 4)).detach().cpu()],
        "std": [float(x) for x in value.float().std(dim=(0, 2, 3, 4)).detach().cpu()],
    }



def _normalized_shifted_frequency_radius(height: int, width: int) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial FFT dimensions must be positive, got {(height, width)}")
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=1.0, dtype=torch.float64)) / 0.5
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=1.0, dtype=torch.float64)) / 0.5
    return torch.sqrt(fy[:, None].square() + fx[None, :].square())


def _is_conjugate_symmetric_mask(mask: torch.Tensor) -> bool:
    unshifted = torch.fft.ifftshift(mask)
    height, width = unshifted.shape
    neg_y = (-torch.arange(height, device=mask.device)) % height
    neg_x = (-torch.arange(width, device=mask.device)) % width
    mirrored = unshifted.index_select(0, neg_y).index_select(1, neg_x)
    return bool(torch.equal(unshifted, mirrored))


def build_stage1_spatial_frequency_mask(
    height: int,
    width: int,
    config: Stage1PhiNoiseConfig,
    *,
    device: torch.device | str,
) -> SpatialFrequencyMask:
    radius = _normalized_shifted_frequency_radius(height, width)
    total_count = height * width
    requested_count: int | None = None
    if config.mask_mode is Stage1PhiNoiseMaskMode.AREA_FRACTION:
        requested_count = round(config.area_fraction * total_count)
        requested_count = min(max(requested_count, 1), total_count - 1)
        candidates: list[tuple[int, float]] = []
        for boundary in torch.unique(radius, sorted=True):
            count = int((radius <= boundary).sum().item())
            if count < total_count:
                candidates.append((count, float(boundary.item())))
        if not candidates:
            raise ValueError("Stage-1 Phi-noise area-fraction mask leaves no high-frequency band")
        actual_count, boundary_radius = min(candidates, key=lambda item: (abs(item[0] - requested_count), item[0]))
        mask = radius <= boundary_radius
    else:
        boundary_radius = float(radius.max().item()) / (2**config.alpha)
        mask = radius < boundary_radius
        actual_count = int(mask.sum().item())
        if actual_count == 0:
            raise ValueError(f"Stage-1 Phi-noise official-alpha mask alpha={config.alpha} selects no bins")
        if actual_count >= total_count:
            raise ValueError("Stage-1 Phi-noise official-alpha mask leaves no high-frequency band")

    mask = mask.to(device=device, dtype=torch.bool)
    if not _is_conjugate_symmetric_mask(mask):
        raise RuntimeError("constructed Stage-1 spatial frequency mask is not conjugate symmetric")
    return SpatialFrequencyMask(
        values=mask,
        requested_count=requested_count,
        actual_count=actual_count,
        actual_fraction=actual_count / total_count,
        boundary_radius=boundary_radius,
    )

def construct_stage1_temporal_phi_noise(  # noqa: PLR0915
    gaussian_noise: torch.Tensor,
    source_latent: torch.Tensor,
    config: Stage1PhiNoiseConfig,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, object]]:
    if gaussian_noise.ndim != 5:
        raise ValueError(f"Stage-1 Phi-noise expects [B,C,T,H,W] Gaussian noise, got {tuple(gaussian_noise.shape)}")
    if source_latent.shape != gaussian_noise.shape:
        raise ValueError(
            "Stage-1 Phi-noise source latent shape must match Gaussian noise: "
            f"source={tuple(source_latent.shape)}, noise={tuple(gaussian_noise.shape)}"
        )
    if not gaussian_noise.is_floating_point() or not source_latent.is_floating_point():
        raise TypeError("Stage-1 Phi-noise inputs must be floating-point tensors")

    original_dtype = gaussian_noise.dtype
    original = gaussian_noise
    output = gaussian_noise.float().clone()
    source = source_latent.to(device=output.device, dtype=torch.float32)
    work_noise = output[:, :, 1:] if config.exclude_frame0 else output
    work_source = source[:, :, 1:] if config.exclude_frame0 else source
    time_count = work_noise.shape[2]
    if time_count < 2:
        raise ValueError(f"Stage-1 Phi-noise needs at least 2 latent frames, got {time_count}")
    freq_bins = time_count // 2 + 1
    if config.k >= freq_bins:
        raise ValueError(f"Stage-1 Phi-noise k={config.k} exceeds max rfft bin {freq_bins - 1}")

    low_bins = config.k + 1
    if low_bins >= freq_bins:
        raise ValueError(
            f"Stage-1 Phi-noise k={config.k} leaves no high-frequency temporal band for T={time_count}"
        )
    low_mask = torch.zeros(freq_bins, device=output.device, dtype=torch.bool)
    low_mask[:low_bins] = True
    low = low_mask.view(1, 1, freq_bins, 1, 1)
    high = ~low

    z = work_noise.float()
    v = work_source.float()
    z_fft = torch.fft.rfft(z, dim=2)
    v_fft = torch.fft.rfft(v, dim=2)

    source_mag = v_fft.abs()
    fallback = (source_mag[:, :, :low_bins] < eps)
    phase_source = torch.where(source_mag >= eps, v_fft / source_mag.clamp_min(eps), z_fft / z_fft.abs().clamp_min(eps))
    own_phase = z_fft / z_fft.abs().clamp_min(eps)
    low_phase = phase_source if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE else own_phase
    substituted_fft = torch.where(low, z_fft.abs() * low_phase, z_fft)

    z_low = torch.fft.irfft(torch.where(low, substituted_fft, torch.zeros_like(substituted_fft)), n=time_count, dim=2)
    z_high = torch.fft.irfft(torch.where(high, substituted_fft, torch.zeros_like(substituted_fft)), n=time_count, dim=2)

    if config.energy_scope is Stage1PhiNoiseEnergyScope.GLOBAL:
        reduce_dims = (1, 2, 3, 4)
        view_shape = (-1, 1, 1, 1, 1)
    else:
        reduce_dims = (2,)
        view_shape = None
    e_tot = z.square().sum(dim=reduce_dims)
    e_low = z_low.square().sum(dim=reduce_dims)
    e_high = z_high.square().sum(dim=reduce_dims)
    if bool(torch.any(e_high <= eps)):
        raise ValueError("Stage-1 Phi-noise high-frequency temporal band has zero energy")
    numerator = e_tot - e_low / (config.gamma**2)
    if bool(torch.any(~torch.isfinite(numerator))) or bool(torch.any(numerator <= 0)):
        raise ValueError("Stage-1 Phi-noise energy compensation has a non-positive or non-finite numerator")
    beta = torch.sqrt(numerator / e_high)
    if bool(torch.any(~torch.isfinite(beta))):
        raise ValueError("Stage-1 Phi-noise produced a non-finite beta")
    beta_view = beta.view(view_shape) if view_shape is not None else beta.unsqueeze(2)
    phi_work = z_low / config.gamma + beta_view * z_high
    if bool(torch.any(~torch.isfinite(phi_work))):
        raise RuntimeError("Stage-1 Phi-noise output contains NaN or Inf")

    if config.exclude_frame0:
        output[:, :, 1:] = phi_work
    else:
        output = phi_work
    final = output.to(original_dtype)

    # Phase diagnostics after balancing; positive real scaling preserves phases.
    final_work = final.float()[:, :, 1:] if config.exclude_frame0 else final.float()
    final_fft = torch.fft.rfft(final_work, dim=2)
    unit_final = final_fft / final_fft.abs().clamp_min(eps)
    unit_source = v_fft / v_fft.abs().clamp_min(eps)
    unit_original = z_fft / z_fft.abs().clamp_min(eps)
    low_valid = (source_mag[:, :, :low_bins] >= eps) & (z_fft.abs()[:, :, :low_bins] >= eps)
    if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE and bool(low_valid.any()):
        low_phase_error = (unit_final[:, :, :low_bins][low_valid] - unit_source[:, :, :low_bins][low_valid]).abs().max()
    else:
        low_phase_error = torch.zeros((), device=output.device)
    high_phase_error = (unit_final[:, :, low_bins:] - unit_original[:, :, low_bins:]).abs().max()

    final_energy = phi_work.square().sum(dim=reduce_dims)
    rel_error = (final_energy - e_tot).abs() / e_tot.clamp_min(eps)
    if config.energy_scope is Stage1PhiNoiseEnergyScope.GLOBAL:
        energy_payload: dict[str, object] = {
            "total_energy": [float(x) for x in e_tot.detach().cpu()],
            "low_energy": [float(x) for x in e_low.detach().cpu()],
            "high_energy": [float(x) for x in e_high.detach().cpu()],
            "balanced_energy": [float(x) for x in final_energy.detach().cpu()],
            "relative_energy_error": [float(x) for x in rel_error.detach().cpu()],
            "beta": [float(x) for x in beta.detach().cpu()],
        }
    else:
        energy_payload = {
            "total_energy_summary": _summary(e_tot),
            "low_energy_summary": _summary(e_low),
            "high_energy_summary": _summary(e_high),
            "balanced_energy_summary": _summary(final_energy),
            "relative_energy_error_summary": _summary(rel_error),
            "beta_summary": _summary(beta),
        }

    diagnostics: dict[str, object] = {
        **config.manifest_dict(),
        "input_shape": list(gaussian_noise.shape),
        "source_shape": list(source_latent.shape),
        "fft_dtype": "torch.float32",
        "latent_frames_used": time_count,
        "rfft_bins": freq_bins,
        "low_bin_start": 0,
        "low_bin_end_inclusive": config.k,
        "low_bin_count": low_bins,
        "source_phase_fallback_count": int(fallback.sum().item()) if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE else 0,
        "source_phase_fallback_fraction": (
            float(fallback.float().mean().item()) if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE else 0.0
        ),
        "low_energy_fraction_before": _summary(e_low / e_tot.clamp_min(eps)),
        "low_energy_fraction_after": _summary((e_low / (config.gamma**2)) / e_tot.clamp_min(eps)),
        "gaussian_channel_stats": _per_channel_mean_std(original.float()),
        "output_channel_stats": _per_channel_mean_std(final.float()),
        "gaussian_checksum": tensor_checksum(original),
        "output_checksum": tensor_checksum(final),
        "source_checksum": tensor_checksum(source_latent),
        "max_low_phase_unit_error": float(low_phase_error.detach().cpu()),
        "max_high_phase_unit_error": float(high_phase_error.detach().cpu()),
        **energy_payload,
    }
    return final, diagnostics


def construct_stage1_spatial_phi_noise(  # noqa: PLR0915
    gaussian_noise: torch.Tensor,
    source_latent: torch.Tensor,
    config: Stage1PhiNoiseConfig,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, object]]:
    if gaussian_noise.ndim != 5:
        raise ValueError(f"Stage-1 spatial Phi-noise expects [B,C,T,H,W], got {tuple(gaussian_noise.shape)}")
    if source_latent.shape != gaussian_noise.shape:
        raise ValueError(
            "Stage-1 spatial Phi-noise source latent shape must match Gaussian noise: "
            f"source={tuple(source_latent.shape)}, noise={tuple(gaussian_noise.shape)}"
        )
    if not gaussian_noise.is_floating_point() or not source_latent.is_floating_point():
        raise TypeError("Stage-1 spatial Phi-noise inputs must be floating-point tensors")

    original_dtype = gaussian_noise.dtype
    noise = gaussian_noise.float()
    source = source_latent.to(device=noise.device, dtype=torch.float32)
    batch, channels, frames, height, width = noise.shape
    mask_info = build_stage1_spatial_frequency_mask(height, width, config, device=noise.device)
    low = mask_info.values.view(1, 1, 1, height, width)
    high = ~low

    z_fft = torch.fft.fftshift(torch.fft.fft2(noise, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    v_fft = torch.fft.fftshift(torch.fft.fft2(source, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    z_mag = z_fft.abs()
    v_mag = v_fft.abs()
    z_unit = z_fft / z_mag.clamp_min(eps)
    v_unit = v_fft / v_mag.clamp_min(eps)
    fallback = (v_mag < eps) & low
    low_phase = torch.where(v_mag >= eps, v_unit, z_unit)
    if config.phase_source is Stage1PhiNoisePhaseSource.OWN:
        low_phase = z_unit
    mixed_fft = torch.where(low, z_mag * low_phase, z_fft)

    low_fft = torch.where(low, mixed_fft, torch.zeros_like(mixed_fft))
    high_fft = torch.where(high, mixed_fft, torch.zeros_like(mixed_fft))
    z_low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
    z_high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real

    if config.energy_scope is Stage1PhiNoiseEnergyScope.GLOBAL:
        reduce_dims = (1, 2, 3, 4)
        view_shape = (-1, 1, 1, 1, 1)
    else:
        reduce_dims = (3, 4)
        view_shape = None
    e_tot = noise.square().sum(dim=reduce_dims)
    e_low = z_low.square().sum(dim=reduce_dims)
    e_high = z_high.square().sum(dim=reduce_dims)
    if bool(torch.any(e_high <= eps)):
        raise ValueError("Stage-1 spatial Phi-noise high-frequency band has zero energy")
    numerator = e_tot - e_low / (config.gamma**2)
    if bool(torch.any(~torch.isfinite(numerator))) or bool(torch.any(numerator <= 0)):
        raise ValueError("Stage-1 spatial Phi-noise energy compensation has a non-positive or non-finite numerator")
    beta = torch.sqrt(numerator / e_high)
    if bool(torch.any(~torch.isfinite(beta))):
        raise ValueError("Stage-1 spatial Phi-noise produced a non-finite beta")
    beta_view = beta.view(view_shape) if view_shape is not None else beta.unsqueeze(-1).unsqueeze(-1)
    real = z_low / config.gamma + beta_view * z_high
    if bool(torch.any(~torch.isfinite(real))):
        raise RuntimeError("Stage-1 spatial Phi-noise output contains NaN or Inf")

    output = real.to(original_dtype)
    final_fft = torch.fft.fftshift(torch.fft.fft2(output.float(), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    final_unit = final_fft / final_fft.abs().clamp_min(eps)
    low_valid = (v_mag >= eps) & (z_mag >= eps) & low
    if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE and bool(low_valid.any()):
        low_phase_error = (final_unit[low_valid] - v_unit[low_valid]).abs().max()
    else:
        low_phase_error = torch.zeros((), device=noise.device)
    high_phase_error = (final_unit[high.expand_as(final_unit)] - z_unit[high.expand_as(z_unit)]).abs().max()

    final_energy = real.square().sum(dim=reduce_dims)
    rel_error = (final_energy - e_tot).abs() / e_tot.clamp_min(eps)
    if config.energy_scope is Stage1PhiNoiseEnergyScope.GLOBAL:
        energy_payload: dict[str, object] = {
            "total_energy": [float(x) for x in e_tot.detach().cpu()],
            "low_energy": [float(x) for x in e_low.detach().cpu()],
            "high_energy": [float(x) for x in e_high.detach().cpu()],
            "balanced_energy": [float(x) for x in final_energy.detach().cpu()],
            "relative_energy_error": [float(x) for x in rel_error.detach().cpu()],
            "beta": [float(x) for x in beta.detach().cpu()],
        }
    else:
        energy_payload = {
            "total_energy_summary": _summary(e_tot),
            "low_energy_summary": _summary(e_low),
            "high_energy_summary": _summary(e_high),
            "balanced_energy_summary": _summary(final_energy),
            "relative_energy_error_summary": _summary(rel_error),
            "beta_summary": _summary(beta),
        }

    diagnostics: dict[str, object] = {
        **config.manifest_dict(),
        "input_shape": list(gaussian_noise.shape),
        "source_shape": list(source_latent.shape),
        "fft_dtype": "torch.float32",
        "spatial_height": height,
        "spatial_width": width,
        "requested_count": mask_info.requested_count,
        "actual_count": mask_info.actual_count,
        "actual_fraction": mask_info.actual_fraction,
        "boundary_normalized_radius": mask_info.boundary_radius,
        "source_phase_fallback_count": int(fallback.sum().item()) if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE else 0,
        "source_phase_fallback_fraction": (
            float(fallback.float().mean().item()) if config.phase_source is Stage1PhiNoisePhaseSource.SOURCE else 0.0
        ),
        "low_energy_fraction_before": _summary(e_low / e_tot.clamp_min(eps)),
        "low_energy_fraction_after": _summary((e_low / (config.gamma**2)) / e_tot.clamp_min(eps)),
        "gaussian_channel_stats": _per_channel_mean_std(gaussian_noise.float()),
        "output_channel_stats": _per_channel_mean_std(output.float()),
        "gaussian_checksum": tensor_checksum(gaussian_noise),
        "output_checksum": tensor_checksum(output),
        "source_checksum": tensor_checksum(source_latent),
        "max_low_phase_unit_error": float(low_phase_error.detach().cpu()),
        "max_high_phase_unit_error": float(high_phase_error.detach().cpu()),
        **energy_payload,
    }
    return output, diagnostics


def construct_stage1_phi_noise(
    gaussian_noise: torch.Tensor,
    source_latent: torch.Tensor,
    config: Stage1PhiNoiseConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    if config.transform is Stage1PhiNoiseTransform.SPATIAL:
        return construct_stage1_spatial_phi_noise(gaussian_noise, source_latent, config)
    return construct_stage1_temporal_phi_noise(gaussian_noise, source_latent, config)


class Stage1PhiNoiser:
    """Gaussian-compatible Stage-1 noiser that manipulates only target video noise."""

    def __init__(
        self,
        *,
        generator: torch.Generator,
        video_tools: VideoLatentTools,
        source_latent: torch.Tensor,
        config: Stage1PhiNoiseConfig,
    ) -> None:
        expected = video_tools.target_shape.to_torch_shape()
        if tuple(source_latent.shape) != tuple(expected):
            raise ValueError(f"Stage-1 Phi source latent shape {tuple(source_latent.shape)} != target {tuple(expected)}")
        self.generator = generator
        self.video_tools = video_tools
        self.source_latent = source_latent
        self.config = config
        self._done = False

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        target_count = self.video_tools.target_shape.token_count()
        target_channels = self.video_tools.target_shape.channels
        is_video = (
            not self._done
            and latent_state.latent.ndim == 3
            and latent_state.latent.shape[1] >= target_count
            and latent_state.latent.shape[2] == target_channels
        )
        noise = torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        if is_video:
            target_noise = self.video_tools.patchifier.unpatchify(
                noise[:, :target_count], output_shape=self.video_tools.target_shape
            )
            transformed, diagnostics = construct_stage1_phi_noise(
                target_noise,
                self.source_latent.to(device=target_noise.device, dtype=target_noise.dtype),
                self.config,
            )
            noise = noise.clone()
            noise[:, :target_count] = self.video_tools.patchifier.patchify(transformed)
            self._done = True
            if self.config.diagnostics_sink is not None:
                source_unchanged = 0.0
                if noise.shape[1] > target_count:
                    # Source/reference suffix stays equal to the original Gaussian draw by construction.
                    source_unchanged = 0.0
                diagnostics["target_token_count"] = target_count
                diagnostics["suffix_token_count"] = int(noise.shape[1] - target_count)
                diagnostics["max_abs_non_target_noise_delta"] = source_unchanged
                self.config.diagnostics_sink.append(diagnostics)
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1.0 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))


def make_stage1_noiser(
    *,
    generator: torch.Generator,
    video_tools: VideoLatentTools,
    source_latent: torch.Tensor | None,
    config: Stage1PhiNoiseConfig | None,
) -> Noiser:
    if config is None or not config.enabled:
        return GaussianNoiser(generator)
    if source_latent is None:
        raise ValueError("Stage-1 Phi-noise requires a source latent")
    return Stage1PhiNoiser(generator=generator, video_tools=video_tools, source_latent=source_latent, config=config)


__all__ = [
    "Stage1PhiNoiseConfig",
    "Stage1PhiNoiseEnergyScope",
    "Stage1PhiNoiseMaskMode",
    "Stage1PhiNoisePhaseSource",
    "Stage1PhiNoiseTransform",
    "SpatialFrequencyMask",
    "build_stage1_spatial_frequency_mask",
    "construct_stage1_phi_noise",
    "construct_stage1_spatial_phi_noise",
    "construct_stage1_temporal_phi_noise",
    "load_stage1_phi_source_latent",
    "make_stage1_noiser",
    "tensor_checksum",
]
