"""Stage-2 target-video noise manipulation for phase-preserving initialization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ltx_core.components.noisers import GaussianNoiser, Noiser
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState


class Stage2NoiseMode(str, Enum):
    GAUSSIAN = "gaussian"
    PHI = "phi"
    RAW_REFERENCE_COEFFICIENTS = "raw_reference_coefficients"
    MATCHED_REFERENCE_COEFFICIENTS = "matched_reference_coefficients"


class Stage2NoiseTransform(str, Enum):
    SPATIAL = "spatial"


class Stage2NoiseMaskMode(str, Enum):
    AREA_FRACTION = "area_fraction"
    OFFICIAL_ALPHA = "official_alpha"


class Stage2NoisePhaseSource(str, Enum):
    STAGE1 = "stage1"


@dataclass
class Stage2NoiseBundle:
    """Replayable raw Gaussian draws used to initialize every Stage-2 modality."""

    target_video: torch.Tensor | None = None
    source_video: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def checksums(self) -> dict[str, str | None]:
        return {
            name: _tensor_checksum(value) if value is not None else None
            for name, value in (
                ("target_video", self.target_video),
                ("source_video", self.source_video),
                ("audio", self.audio),
            )
        }


def save_stage2_noise_bundle(path: str | Path, bundle: Stage2NoiseBundle) -> None:
    bundle_path = Path(path).expanduser().resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in (
            ("target_video", bundle.target_video),
            ("source_video", bundle.source_video),
            ("audio", bundle.audio),
        )
        if value is not None
    }
    if "target_video" not in tensors or "audio" not in tensors:
        raise ValueError("Stage-2 noise bundle requires target_video and audio tensors")
    save_file(tensors, bundle_path)
    manifest = {
        "version": 1,
        "provenance": bundle.provenance,
        "shapes": {name: list(value.shape) for name, value in tensors.items()},
        "dtypes": {name: str(value.dtype) for name, value in tensors.items()},
        "checksums": bundle.checksums(),
    }
    bundle_path.with_suffix(bundle_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load_stage2_noise_bundle(path: str | Path) -> Stage2NoiseBundle:
    bundle_path = Path(path).expanduser().resolve()
    manifest_path = bundle_path.with_suffix(bundle_path.suffix + ".json")
    tensors = load_file(bundle_path, device="cpu")
    manifest = json.loads(manifest_path.read_text())
    bundle = Stage2NoiseBundle(
        target_video=tensors.get("target_video"),
        source_video=tensors.get("source_video"),
        audio=tensors.get("audio"),
        provenance=dict(manifest.get("provenance", {})),
    )
    if bundle.target_video is None or bundle.audio is None:
        raise ValueError("Stage-2 noise bundle requires target_video and audio tensors")
    if bundle.checksums() != manifest.get("checksums"):
        raise ValueError(f"Stage-2 noise bundle checksum mismatch: {bundle_path}")
    return bundle


@dataclass(frozen=True)
class Stage2NoiseConfig:
    """Configuration for Stage-2 target-video noise construction."""

    mode: Stage2NoiseMode | str = Stage2NoiseMode.GAUSSIAN
    transform: Stage2NoiseTransform | str = Stage2NoiseTransform.SPATIAL
    mask_mode: Stage2NoiseMaskMode | str = Stage2NoiseMaskMode.AREA_FRACTION
    area_fraction: float = 0.05
    alpha: int = 3
    gamma: float = 5.0
    phase_source: Stage2NoisePhaseSource | str = Stage2NoisePhaseSource.STAGE1
    diagnostics_sink: list[dict[str, object]] | None = field(default=None, compare=False, repr=False)
    noise_bundle: Stage2NoiseBundle | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Stage2NoiseMode(self.mode))
        object.__setattr__(self, "transform", Stage2NoiseTransform(self.transform))
        object.__setattr__(self, "mask_mode", Stage2NoiseMaskMode(self.mask_mode))
        object.__setattr__(self, "phase_source", Stage2NoisePhaseSource(self.phase_source))
        if not 0.0 < self.area_fraction < 1.0:
            raise ValueError(f"Stage-2 noise area_fraction must be in (0,1), got {self.area_fraction}")
        if self.alpha <= 0:
            raise ValueError(f"Stage-2 noise alpha must be positive, got {self.alpha}")
        if not math.isfinite(self.gamma) or self.gamma < 1.0:
            raise ValueError(f"Stage-2 noise gamma must be finite and >= 1, got {self.gamma}")

    def manifest_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "transform": self.transform.value,
            "mask_mode": self.mask_mode.value,
            "area_fraction": self.area_fraction,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "phase_source": self.phase_source.value,
        }


@dataclass(frozen=True)
class SpatialFrequencyMask:
    values: torch.Tensor
    requested_count: int | None
    actual_count: int
    actual_fraction: float
    boundary_radius: float


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


def build_spatial_frequency_mask(
    height: int,
    width: int,
    config: Stage2NoiseConfig,
    *,
    device: torch.device | str,
) -> SpatialFrequencyMask:
    """Build a centered hard radial mask with deterministic complete-ring selection."""

    radius = _normalized_shifted_frequency_radius(height, width)
    total_count = height * width
    requested_count: int | None = None
    if config.mask_mode is Stage2NoiseMaskMode.AREA_FRACTION:
        requested_count = round(config.area_fraction * total_count)
        requested_count = min(max(requested_count, 1), total_count - 1)
        candidates: list[tuple[int, float]] = []
        for boundary in torch.unique(radius, sorted=True):
            count = int((radius <= boundary).sum().item())
            if count < total_count:
                candidates.append((count, float(boundary.item())))
        if not candidates:
            raise ValueError("area-fraction mask leaves no high-frequency band")
        actual_count, boundary_radius = min(
            candidates,
            key=lambda item: (abs(item[0] - requested_count), item[0]),
        )
        mask = radius <= boundary_radius
    else:
        boundary_radius = float(radius.max().item()) / (2**config.alpha)
        mask = radius < boundary_radius
        actual_count = int(mask.sum().item())
        if actual_count == 0:
            raise ValueError(f"official-alpha mask alpha={config.alpha} selects no bins for grid {height}x{width}")
        if actual_count >= total_count:
            raise ValueError("official-alpha mask leaves no high-frequency band")

    mask = mask.to(device=device, dtype=torch.bool)
    if not _is_conjugate_symmetric_mask(mask):
        raise RuntimeError("constructed spatial frequency mask is not conjugate symmetric")
    return SpatialFrequencyMask(
        values=mask,
        requested_count=requested_count,
        actual_count=actual_count,
        actual_fraction=actual_count / total_count,
        boundary_radius=boundary_radius,
    )


def _sum_spectral_energy(value: torch.Tensor) -> torch.Tensor:
    return value.abs().square().sum(dim=(1, 2, 3, 4))


def _lag_one_correlations(value: torch.Tensor) -> dict[str, list[float]]:
    centered = value.float() - value.float().mean(dim=(1, 2, 3, 4), keepdim=True)
    variance = centered.square().mean(dim=(1, 2, 3, 4)).clamp_min(1e-12)
    batch_size = value.shape[0]
    horizontal = (
        (centered[..., :, :-1] * centered[..., :, 1:]).mean(dim=(1, 2, 3, 4)) / variance
        if value.shape[-1] > 1
        else torch.zeros(batch_size, device=value.device)
    )
    vertical = (
        (centered[..., :-1, :] * centered[..., 1:, :]).mean(dim=(1, 2, 3, 4)) / variance
        if value.shape[-2] > 1
        else torch.zeros(batch_size, device=value.device)
    )
    return {
        "horizontal": [float(item) for item in horizontal.detach().cpu()],
        "vertical": [float(item) for item in vertical.detach().cpu()],
    }


def _tensor_checksum(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def draw_or_replay_stage2_noise(
    *,
    bundle: Stage2NoiseBundle | None,
    role: str,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if role not in {"target_video", "source_video", "audio"}:
        raise ValueError(f"unknown Stage-2 noise role: {role}")
    existing = getattr(bundle, role) if bundle is not None else None
    if existing is not None:
        if tuple(existing.shape) != tuple(shape):
            raise ValueError(
                f"Stage-2 replay noise shape mismatch for {role}: saved={tuple(existing.shape)}, requested={tuple(shape)}"
            )
        return existing.to(device=device, dtype=dtype)
    noise = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    if bundle is not None:
        setattr(bundle, role, noise.detach().to(device="cpu").contiguous())
    return noise


class Stage2RoleNoiser:
    """Gaussian noiser backed by one named tensor in a Stage2NoiseBundle."""

    def __init__(self, *, generator: torch.Generator, bundle: Stage2NoiseBundle | None, role: str) -> None:
        self.generator = generator
        self.bundle = bundle
        self.role = role

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        noise = draw_or_replay_stage2_noise(
            bundle=self.bundle,
            role=self.role,
            shape=latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1.0 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))


def construct_stage2_video_noise(  # noqa: PLR0915
    gaussian_noise: torch.Tensor,
    reference_latent: torch.Tensor,
    config: Stage2NoiseConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Construct energy-balanced Stage-2 video noise from an existing Gaussian draw."""

    if gaussian_noise.ndim != 5:
        raise ValueError(f"Stage-2 video noise must have shape [B,C,F,H,W], got {gaussian_noise.shape}")
    if gaussian_noise.shape != reference_latent.shape:
        raise ValueError(
            "Stage-2 phase reference shape must exactly match Gaussian noise: "
            f"reference={tuple(reference_latent.shape)}, noise={tuple(gaussian_noise.shape)}"
        )
    if not gaussian_noise.is_floating_point() or not reference_latent.is_floating_point():
        raise TypeError("Stage-2 Gaussian noise and phase reference must be floating-point tensors")

    original_dtype = gaussian_noise.dtype
    noise = gaussian_noise.float()
    reference = reference_latent.to(device=noise.device, dtype=torch.float32)
    if config.mode is Stage2NoiseMode.GAUSSIAN:
        noise_fft = torch.fft.fft2(noise, dim=(-2, -1), norm="ortho")
        target_energy = _sum_spectral_energy(noise_fft)
        batch_size = noise.shape[0]
        zeros = [0.0] * batch_size
        ones = [1.0] * batch_size
        means = [float(item) for item in noise.mean(dim=(1, 2, 3, 4)).detach().cpu()]
        stds = [float(item) for item in noise.std(dim=(1, 2, 3, 4)).detach().cpu()]
        energies = [float(item) for item in target_energy.detach().cpu()]
        correlations = _lag_one_correlations(noise)
        checksum = _tensor_checksum(gaussian_noise)
        return gaussian_noise, {
            **config.manifest_dict(),
            "fft_dtype": str(noise.dtype),
            "requested_count": None,
            "actual_count": 0,
            "actual_fraction": 0.0,
            "boundary_normalized_radius": None,
            "target_energy": energies,
            "gaussian_low_energy": zeros,
            "modified_low_energy": zeros,
            "gaussian_high_energy": energies,
            "balanced_energy": energies,
            "relative_energy_error": zeros,
            "beta": ones,
            "max_imaginary_residual": zeros,
            "relative_imaginary_residual": zeros,
            "gaussian_mean": means,
            "gaussian_std": stds,
            "output_mean": means,
            "output_std": stds,
            "gaussian_lag1_correlation": correlations,
            "output_lag1_correlation": correlations,
            "gaussian_checksum": checksum,
            "output_checksum": checksum,
            "reference_checksum": _tensor_checksum(reference_latent),
        }

    height, width = noise.shape[-2:]
    mask_info = build_spatial_frequency_mask(height, width, config, device=noise.device)
    low_mask = mask_info.values.view(1, 1, 1, height, width)
    high_mask = ~low_mask

    noise_fft = torch.fft.fftshift(torch.fft.fft2(noise, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    reference_fft = torch.fft.fftshift(torch.fft.fft2(reference, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    gaussian_low_energy = _sum_spectral_energy(noise_fft * low_mask)
    gaussian_high_energy = _sum_spectral_energy(noise_fft * high_mask)
    target_energy = _sum_spectral_energy(noise_fft)

    if config.mode is Stage2NoiseMode.PHI:
        reference_phase = torch.angle(reference_fft)
        phase_substituted = torch.polar(torch.abs(noise_fft), reference_phase)
        mixed_fft = torch.where(low_mask, phase_substituted, noise_fft)
    elif config.mode is Stage2NoiseMode.RAW_REFERENCE_COEFFICIENTS:
        mixed_fft = torch.where(low_mask, reference_fft, noise_fft)
    else:
        reference_low_energy = _sum_spectral_energy(reference_fft * low_mask)
        if bool(torch.any(reference_low_energy <= 1e-12)):
            raise ValueError("cannot energy-match a zero-energy Stage-1 low-frequency band")
        match_scale = torch.sqrt(gaussian_low_energy / reference_low_energy).view(-1, 1, 1, 1, 1)
        matched_reference_fft = reference_fft * match_scale
        mixed_fft = torch.where(low_mask, matched_reference_fft, noise_fft)

    modified_low_energy = _sum_spectral_energy(mixed_fft * low_mask)
    if bool(torch.any(gaussian_high_energy <= 1e-12)):
        raise ValueError("Stage-2 noise mask has zero Gaussian high-frequency energy")
    compensation_numerator = target_energy - modified_low_energy / (config.gamma**2)
    if bool(torch.any(~torch.isfinite(compensation_numerator))) or bool(torch.any(compensation_numerator <= 0)):
        raise ValueError("Stage-2 spectral energy compensation has a non-positive or non-finite numerator")
    beta = torch.sqrt(compensation_numerator / gaussian_high_energy)
    if bool(torch.any(~torch.isfinite(beta))):
        raise ValueError("Stage-2 spectral energy compensation produced a non-finite beta")
    scale = low_mask.to(noise_fft.real.dtype) / config.gamma + high_mask.to(noise_fft.real.dtype) * beta.view(
        -1, 1, 1, 1, 1
    )
    balanced_fft = mixed_fft * scale
    balanced_energy = _sum_spectral_energy(balanced_fft)

    reconstructed = torch.fft.ifft2(torch.fft.ifftshift(balanced_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    real = reconstructed.real
    max_imaginary = reconstructed.imag.abs().reshape(reconstructed.shape[0], -1).amax(dim=1)
    real_rms = real.square().mean(dim=(1, 2, 3, 4)).sqrt().clamp_min(1e-12)
    imaginary_ratio = max_imaginary / real_rms
    if bool(torch.any(~torch.isfinite(imaginary_ratio))) or bool(torch.any(imaginary_ratio > 1e-4)):
        raise RuntimeError(
            "Stage-2 inverse FFT is not sufficiently real-valued: "
            f"relative residuals={imaginary_ratio.detach().cpu().tolist()}"
        )

    output = real.to(original_dtype)
    relative_energy_error = (balanced_energy - target_energy).abs() / target_energy.clamp_min(1e-12)
    diagnostics: dict[str, object] = {
        **config.manifest_dict(),
        "fft_dtype": str(noise.dtype),
        "requested_count": mask_info.requested_count,
        "actual_count": mask_info.actual_count,
        "actual_fraction": mask_info.actual_fraction,
        "boundary_normalized_radius": mask_info.boundary_radius,
        "target_energy": [float(item) for item in target_energy.detach().cpu()],
        "gaussian_low_energy": [float(item) for item in gaussian_low_energy.detach().cpu()],
        "modified_low_energy": [float(item) for item in modified_low_energy.detach().cpu()],
        "gaussian_high_energy": [float(item) for item in gaussian_high_energy.detach().cpu()],
        "balanced_energy": [float(item) for item in balanced_energy.detach().cpu()],
        "relative_energy_error": [float(item) for item in relative_energy_error.detach().cpu()],
        "beta": [float(item) for item in beta.detach().cpu()],
        "max_imaginary_residual": [float(item) for item in max_imaginary.detach().cpu()],
        "relative_imaginary_residual": [float(item) for item in imaginary_ratio.detach().cpu()],
        "gaussian_mean": [float(item) for item in noise.mean(dim=(1, 2, 3, 4)).detach().cpu()],
        "gaussian_std": [float(item) for item in noise.std(dim=(1, 2, 3, 4)).detach().cpu()],
        "output_mean": [float(item) for item in real.mean(dim=(1, 2, 3, 4)).detach().cpu()],
        "output_std": [float(item) for item in real.std(dim=(1, 2, 3, 4)).detach().cpu()],
        "gaussian_lag1_correlation": _lag_one_correlations(noise),
        "output_lag1_correlation": _lag_one_correlations(real),
        "gaussian_checksum": _tensor_checksum(gaussian_noise),
        "output_checksum": _tensor_checksum(output),
        "reference_checksum": _tensor_checksum(reference_latent),
    }
    return output, diagnostics


class Stage2TargetNoiser:
    """Gaussian-compatible noiser that observes or manipulates the target-video prefix once."""

    def __init__(
        self,
        *,
        generator: torch.Generator,
        video_tools: VideoLatentTools,
        reference_latent: torch.Tensor,
        config: Stage2NoiseConfig,
    ) -> None:
        expected_shape = video_tools.target_shape.to_torch_shape()
        if reference_latent.shape != expected_shape:
            raise ValueError(
                f"Stage-2 reference latent shape {reference_latent.shape} does not match target {expected_shape}"
            )
        self.generator = generator
        self.video_tools = video_tools
        self.reference_latent = reference_latent
        self.config = config
        self._target_was_manipulated = False

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        target_count = self.video_tools.target_shape.token_count()
        target_channels = self.video_tools.target_shape.channels
        is_target_video_state = (
            not self._target_was_manipulated
            and latent_state.latent.ndim == 3
            and latent_state.latent.shape[1] >= target_count
            and latent_state.latent.shape[2] == target_channels
        )
        if is_target_video_state:
            has_source = latent_state.latent.shape[1] > target_count
            bundle = self.config.noise_bundle
            draw_combined = bundle is None or (
                bundle.target_video is None and (not has_source or bundle.source_video is None)
            )
            if draw_combined:
                noise = torch.randn(
                    *latent_state.latent.shape,
                    device=latent_state.latent.device,
                    dtype=latent_state.latent.dtype,
                    generator=self.generator,
                )
                if bundle is not None:
                    bundle.target_video = noise[:, :target_count].detach().to(device="cpu").contiguous()
                    if has_source:
                        bundle.source_video = noise[:, target_count:].detach().to(device="cpu").contiguous()
            else:
                target_noise = draw_or_replay_stage2_noise(
                    bundle=bundle,
                    role="target_video",
                    shape=latent_state.latent[:, :target_count].shape,
                    device=latent_state.latent.device,
                    dtype=latent_state.latent.dtype,
                    generator=self.generator,
                )
                if has_source:
                    source_noise = draw_or_replay_stage2_noise(
                        bundle=bundle,
                        role="source_video",
                        shape=latent_state.latent[:, target_count:].shape,
                        device=latent_state.latent.device,
                        dtype=latent_state.latent.dtype,
                        generator=self.generator,
                    )
                    noise = torch.cat((target_noise, source_noise), dim=1)
                else:
                    noise = target_noise
            unpatchified_noise = self.video_tools.patchifier.unpatchify(
                noise[:, :target_count], output_shape=self.video_tools.target_shape
            )
            transformed_noise, diagnostics = construct_stage2_video_noise(
                unpatchified_noise,
                self.reference_latent,
                self.config,
            )
            noise = noise.clone()
            noise[:, :target_count] = self.video_tools.patchifier.patchify(transformed_noise)
            self._target_was_manipulated = True
            if self.config.diagnostics_sink is not None:
                self.config.diagnostics_sink.append(diagnostics)
        else:
            noise = draw_or_replay_stage2_noise(
                bundle=self.config.noise_bundle,
                role="audio",
                shape=latent_state.latent.shape,
                device=latent_state.latent.device,
                dtype=latent_state.latent.dtype,
                generator=self.generator,
            )

        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1.0 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))


def make_stage2_noiser(
    *,
    generator: torch.Generator,
    video_tools: VideoLatentTools,
    reference_latent: torch.Tensor,
    config: Stage2NoiseConfig,
) -> Noiser:
    if config.mode is Stage2NoiseMode.GAUSSIAN and config.diagnostics_sink is None and config.noise_bundle is None:
        return GaussianNoiser(generator)
    return Stage2TargetNoiser(
        generator=generator,
        video_tools=video_tools,
        reference_latent=reference_latent,
        config=config,
    )


__all__ = [
    "SpatialFrequencyMask",
    "Stage2NoiseConfig",
    "Stage2NoiseBundle",
    "Stage2NoiseMaskMode",
    "Stage2NoiseMode",
    "Stage2NoisePhaseSource",
    "Stage2NoiseTransform",
    "Stage2TargetNoiser",
    "Stage2RoleNoiser",
    "build_spatial_frequency_mask",
    "construct_stage2_video_noise",
    "draw_or_replay_stage2_noise",
    "load_stage2_noise_bundle",
    "make_stage2_noiser",
    "save_stage2_noise_bundle",
]
