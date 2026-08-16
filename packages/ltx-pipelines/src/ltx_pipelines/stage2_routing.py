"""Stage-2 prediction-space routing for image and IC-LoRA video branches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import runtime_lora_scale
from ltx_core.model.transformer import X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import create_noised_state, post_process_latent, state_with_conditionings
from ltx_pipelines.utils.types import Denoiser


class Stage2BranchMode(str, Enum):
    LEGACY = "legacy"
    IMAGE = "image"
    VIDEO = "video"
    GLOBAL = "global"
    SPATIAL = "spatial"

    @property
    def is_routed(self) -> bool:
        return self is not Stage2BranchMode.LEGACY

    @property
    def needs_both_branches(self) -> bool:
        return self in (Stage2BranchMode.GLOBAL, Stage2BranchMode.SPATIAL)


def downsample_semantic_mask_to_target_tokens(mask: torch.Tensor, tools: VideoLatentTools) -> torch.Tensor:
    """Convert a pixel-space semantic mask to ``(B, target_tokens)`` weights.

    Spatial dimensions use area averaging. Causal temporal groups use mean
    pooling so soft feathering is retained rather than expanded by max pooling.
    """

    shape = tools.target_shape
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError(f"routing mask must have shape (B,1,F,H,W), got {tuple(mask.shape)}")
    if mask.shape[0] not in (1, shape.batch):
        raise ValueError(f"routing mask batch must be 1 or {shape.batch}, got {mask.shape[0]}")
    batch, _, pixel_frames, _, _ = mask.shape
    spatial = torch.nn.functional.interpolate(
        mask.float().movedim(2, 1).reshape(batch * pixel_frames, 1, mask.shape[3], mask.shape[4]),
        size=(shape.height, shape.width),
        mode="area",
    ).reshape(batch, pixel_frames, 1, shape.height, shape.width).movedim(1, 2)
    first = spatial[:, :, :1]
    if shape.frames == 1:
        latent_mask = first
    else:
        remaining_pixels = pixel_frames - 1
        remaining_latents = shape.frames - 1
        if remaining_pixels <= 0 or remaining_pixels % remaining_latents != 0:
            raise ValueError(
                f"routing mask has {pixel_frames} frames, incompatible with causal latent frames {shape.frames}"
            )
        group = remaining_pixels // remaining_latents
        rest = spatial[:, :, 1:].reshape(
            batch, 1, remaining_latents, group, shape.height, shape.width
        ).mean(dim=3)
        latent_mask = torch.cat((first, rest), dim=2)
    return latent_mask.clamp(0, 1).permute(0, 2, 3, 4, 1).reshape(batch, -1)


def route_stage2_predictions(
    image_x0: torch.Tensor,
    video_x0: torch.Tensor,
    *,
    mode: Stage2BranchMode | str,
    video_mix: float = 0.5,
    routing_mask: torch.Tensor | None = None,
    dress_video_contribution: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend target-token X0 predictions and return ``(prediction, video_weight)``."""

    mode = Stage2BranchMode(mode)
    if image_x0.shape != video_x0.shape:
        raise ValueError(f"branch prediction shapes differ: {image_x0.shape} vs {video_x0.shape}")
    if mode is Stage2BranchMode.GLOBAL:
        if not 0.0 <= video_mix <= 1.0:
            raise ValueError(f"video_mix must be in [0,1], got {video_mix}")
        video_weight = torch.full(
            (*image_x0.shape[:2], 1), video_mix, device=image_x0.device, dtype=image_x0.dtype
        )
    elif mode is Stage2BranchMode.SPATIAL:
        if routing_mask is None:
            raise ValueError("routing_mask is required for spatial Stage-2 routing")
        if dress_video_contribution is None or not 0.0 <= dress_video_contribution <= 1.0:
            raise ValueError("dress_video_contribution must be in [0,1] for spatial routing")
        mask = routing_mask.to(device=image_x0.device, dtype=image_x0.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape != (*image_x0.shape[:2], 1):
            raise ValueError(f"routing mask shape {mask.shape} does not match tokens {image_x0.shape[:2]}")
        video_weight = 1.0 - mask * (1.0 - dress_video_contribution)
    else:
        raise ValueError(f"prediction blending requires global or spatial mode, got {mode.value}")
    routed = image_x0 * (1.0 - video_weight) + video_x0 * video_weight
    return routed, video_weight


def tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _cache_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def save_stage2_input_cache(
    path: str | Path,
    *,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    cache_path = Path(path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "video_latent": video_latent.detach().to(device="cpu").contiguous(),
        "audio_latent": audio_latent.detach().to(device="cpu").contiguous(),
    }
    save_file(tensors, cache_path)
    manifest = {
        **metadata,
        "video_shape": list(video_latent.shape),
        "audio_shape": list(audio_latent.shape),
        "video_checksum": tensor_checksum(video_latent),
        "audio_checksum": tensor_checksum(audio_latent),
    }
    _cache_manifest_path(cache_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_stage2_input_cache(
    path: str | Path,
    *,
    expected_metadata: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    cache_path = Path(path).expanduser().resolve()
    manifest_path = _cache_manifest_path(cache_path)
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-2 cache requires {cache_path} and {manifest_path}")
    metadata = json.loads(manifest_path.read_text())
    if expected_metadata is not None:
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            details = ", ".join(f"{key}: cached={old!r}, requested={new!r}" for key, (old, new) in mismatches.items())
            raise ValueError(f"Stage-2 input cache metadata mismatch: {details}")
    tensors = load_file(cache_path, device="cpu")
    if set(tensors) != {"video_latent", "audio_latent"}:
        raise ValueError(f"invalid Stage-2 cache tensors: {sorted(tensors)}")
    video_latent = tensors["video_latent"]
    audio_latent = tensors["audio_latent"]
    shapes_mismatch = list(video_latent.shape) != metadata.get("video_shape") or list(
        audio_latent.shape
    ) != metadata.get("audio_shape")
    if shapes_mismatch:
        raise ValueError("Stage-2 input cache tensor shapes do not match its manifest")
    if tensor_checksum(video_latent) != metadata.get("video_checksum"):
        raise ValueError("Stage-2 input cache video checksum mismatch")
    if tensor_checksum(audio_latent) != metadata.get("audio_checksum"):
        raise ValueError("Stage-2 input cache audio checksum mismatch")
    return video_latent, audio_latent, metadata


def _weighted_rms(delta: torch.Tensor, weights: torch.Tensor) -> float:
    weights = weights.to(device=delta.device, dtype=torch.float32).unsqueeze(-1)
    numerator = (delta.float().square() * weights).sum()
    denominator = weights.sum() * delta.shape[-1]
    if denominator.item() == 0:
        return 0.0
    return float(torch.sqrt(numerator / denominator).cpu())


class Stage2PredictionRecorder:
    """Stream target-only branch predictions and summary metrics to disk."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, object], routing_mask: torch.Tensor | None) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.routing_mask = routing_mask.detach().cpu() if routing_mask is not None else None
        self.manifest = {**metadata, "steps": []}
        if self.routing_mask is not None:
            save_file({"routing_mask": self.routing_mask.contiguous()}, self.output_dir / "routing_mask.safetensors")

    def record(
        self,
        *,
        step_index: int,
        sigma: float,
        shared_input: torch.Tensor,
        routed_x0: torch.Tensor,
        image_x0: torch.Tensor | None,
        video_x0: torch.Tensor | None,
        video_weight: torch.Tensor,
    ) -> None:
        tensors = {
            "shared_input": shared_input.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "routed_x0": routed_x0.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "video_weight": video_weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        }
        if image_x0 is not None:
            tensors["image_x0"] = image_x0.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if video_x0 is not None:
            tensors["video_x0"] = video_x0.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        filename = f"step_{step_index:03d}.safetensors"
        save_file(tensors, self.output_dir / filename, metadata={"step": str(step_index), "sigma": repr(sigma)})
        row: dict[str, object] = {"step": step_index, "sigma": sigma, "file": filename}
        if image_x0 is not None and video_x0 is not None:
            delta = video_x0 - image_x0
            row["branch_delta_rms"] = float(delta.float().square().mean().sqrt().cpu())
            if self.routing_mask is not None:
                mask = self.routing_mask.to(delta.device)
                row["branch_delta_rms_inside_mask"] = _weighted_rms(delta, mask)
                row["branch_delta_rms_outside_mask"] = _weighted_rms(delta, 1.0 - mask)
        self.manifest["steps"].append(row)
        (self.output_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")


def _noise_appended_tokens(
    state: LatentState,
    *,
    target_token_count: int,
    noise_scale: float,
    generator: torch.Generator,
) -> LatentState:
    if state.latent.shape[1] == target_token_count:
        return state
    latent = state.latent.clone()
    suffix = slice(target_token_count, None)
    noise = torch.randn(
        latent[:, suffix].shape,
        device=latent.device,
        dtype=latent.dtype,
        generator=generator,
    )
    scaled_mask = state.denoise_mask[:, suffix] * noise_scale
    latent[:, suffix] = noise * scaled_mask + latent[:, suffix] * (1.0 - scaled_mask)
    return replace(state, latent=latent)


def run_routed_stage2(  # noqa: PLR0913,PLR0915
    *,
    transformer: X0Model,
    sigmas: torch.Tensor,
    video_tools: VideoLatentTools,
    audio_tools: AudioLatentTools,
    image_conditionings: list[ConditioningItem],
    video_conditionings: list[ConditioningItem],
    initial_video_latent: torch.Tensor,
    initial_audio_latent: torch.Tensor,
    image_denoiser: Denoiser,
    video_denoiser: Denoiser,
    mode: Stage2BranchMode | str,
    noise_seed: int,
    video_mix: float = 0.5,
    routing_mask: torch.Tensor | None = None,
    dress_video_contribution: float | None = None,
    recorder: Stage2PredictionRecorder | None = None,
    video_denoise_mask: torch.Tensor | None = None,
) -> tuple[LatentState, LatentState]:
    """Run Stage 2 with one shared video trajectory and branch-local audio."""

    mode = Stage2BranchMode(mode)
    if not mode.is_routed:
        raise ValueError("run_routed_stage2 cannot run legacy mode")
    target_count = video_tools.target_shape.token_count()
    noise_scale = float(sigmas[0].item())
    target_generator = torch.Generator(device=initial_video_latent.device).manual_seed(noise_seed)
    source_generator = torch.Generator(device=initial_video_latent.device).manual_seed(noise_seed + 1)
    audio_generator = torch.Generator(device=initial_audio_latent.device).manual_seed(noise_seed + 2)

    image_state = create_noised_state(
        tools=video_tools,
        conditionings=image_conditionings,
        noiser=GaussianNoiser(target_generator),
        dtype=initial_video_latent.dtype,
        device=initial_video_latent.device,
        noise_scale=noise_scale,
        denoise_mask=video_denoise_mask,
        initial_latent=initial_video_latent,
    )
    video_state = state_with_conditionings(image_state.clone(), video_conditionings, video_tools)
    video_state = _noise_appended_tokens(
        video_state,
        target_token_count=target_count,
        noise_scale=noise_scale,
        generator=source_generator,
    )
    shared_state = image_state
    base_audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=[],
        noiser=GaussianNoiser(audio_generator),
        dtype=initial_audio_latent.dtype,
        device=initial_audio_latent.device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )
    image_audio_state = base_audio_state.clone()
    video_audio_state = base_audio_state.clone()
    stepper = EulerDiffusionStep()

    for step_index in tqdm(range(sigmas.numel() - 1), desc="Stage 2 routed denoising"):
        target_latent = shared_state.latent
        image_state = replace(image_state, latent=target_latent)
        video_latent = video_state.latent.clone()
        video_latent[:, :target_count] = target_latent
        video_state = replace(video_state, latent=video_latent)

        image_x0 = None
        video_x0 = None
        image_audio_x0 = None
        video_audio_x0 = None
        if mode in (Stage2BranchMode.IMAGE, Stage2BranchMode.GLOBAL, Stage2BranchMode.SPATIAL):
            with runtime_lora_scale(transformer, 0.0):
                image_result, image_audio_result = image_denoiser(
                    transformer, image_state, image_audio_state, sigmas, step_index
                )
            if image_result is None or image_audio_result is None:
                raise RuntimeError("image branch did not return video and audio predictions")
            image_x0 = image_result.denoised[:, :target_count]
            image_audio_x0 = image_audio_result.denoised

        if mode in (Stage2BranchMode.VIDEO, Stage2BranchMode.GLOBAL, Stage2BranchMode.SPATIAL):
            with runtime_lora_scale(transformer, 1.0):
                video_result, video_audio_result = video_denoiser(
                    transformer, video_state, video_audio_state, sigmas, step_index
                )
            if video_result is None or video_audio_result is None:
                raise RuntimeError("video branch did not return video and audio predictions")
            video_x0 = video_result.denoised[:, :target_count]
            video_audio_x0 = video_audio_result.denoised

        if mode is Stage2BranchMode.IMAGE:
            assert image_x0 is not None
            routed_x0 = image_x0
            video_weight = torch.zeros((*routed_x0.shape[:2], 1), device=routed_x0.device, dtype=routed_x0.dtype)
        elif mode is Stage2BranchMode.VIDEO:
            assert video_x0 is not None
            routed_x0 = video_x0
            video_weight = torch.ones((*routed_x0.shape[:2], 1), device=routed_x0.device, dtype=routed_x0.dtype)
        else:
            assert image_x0 is not None
            assert video_x0 is not None
            routed_x0, video_weight = route_stage2_predictions(
                image_x0,
                video_x0,
                mode=mode,
                video_mix=video_mix,
                routing_mask=routing_mask,
                dress_video_contribution=dress_video_contribution,
            )

        if recorder is not None:
            recorder.record(
                step_index=step_index,
                sigma=float(sigmas[step_index].detach().cpu()),
                shared_input=target_latent,
                routed_x0=routed_x0,
                image_x0=image_x0,
                video_x0=video_x0,
                video_weight=video_weight,
            )
        routed_x0 = post_process_latent(routed_x0, shared_state.denoise_mask, shared_state.clean_latent)
        shared_state = replace(
            shared_state,
            latent=stepper.step(shared_state.latent, routed_x0, sigmas, step_index),
        )
        if image_audio_x0 is not None:
            image_audio_x0 = post_process_latent(
                image_audio_x0, image_audio_state.denoise_mask, image_audio_state.clean_latent
            )
            image_audio_state = replace(
                image_audio_state,
                latent=stepper.step(image_audio_state.latent, image_audio_x0, sigmas, step_index),
            )
        if video_audio_x0 is not None:
            video_audio_x0 = post_process_latent(
                video_audio_x0, video_audio_state.denoise_mask, video_audio_state.clean_latent
            )
            video_audio_state = replace(
                video_audio_state,
                latent=stepper.step(video_audio_state.latent, video_audio_x0, sigmas, step_index),
            )

    output_audio = image_audio_state if mode is Stage2BranchMode.IMAGE else video_audio_state
    return video_tools.unpatchify(shared_state), audio_tools.unpatchify(output_audio)
