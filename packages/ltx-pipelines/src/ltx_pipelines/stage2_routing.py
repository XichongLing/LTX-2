"""Stage-2 prediction-space routing for image and IC-LoRA video branches."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from contextlib import nullcontext
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
from ltx_pipelines.stage2_kv import Stage2KVAttentionController
from ltx_pipelines.stage2_noise import (
    Stage2NoiseConfig,
    Stage2RoleNoiser,
    draw_or_replay_stage2_noise,
    make_stage2_noiser,
)
from ltx_pipelines.stage2_reference import Stage2ReferenceAttentionController
from ltx_pipelines.utils.helpers import create_noised_state, post_process_latent, state_with_conditionings
from ltx_pipelines.utils.types import Denoiser


class Stage2BranchMode(str, Enum):
    LEGACY = "legacy"
    IMAGE = "image"
    VIDEO = "video"
    GLOBAL = "global"
    SPATIAL = "spatial"
    DUAL = "dual"

    @property
    def is_routed(self) -> bool:
        return self is not Stage2BranchMode.LEGACY

    @property
    def needs_both_branches(self) -> bool:
        return self in (Stage2BranchMode.GLOBAL, Stage2BranchMode.SPATIAL, Stage2BranchMode.DUAL)


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


def downsample_semantic_mask_to_hard_target_tokens(mask: torch.Tensor, tools: VideoLatentTools) -> torch.Tensor:
    """Conservatively map a semantic mask to binary Stage-2 target tokens."""

    shape = tools.target_shape
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError(f"routing mask must have shape (B,1,F,H,W), got {tuple(mask.shape)}")
    if mask.shape[0] not in (1, shape.batch):
        raise ValueError(f"routing mask batch must be 1 or {shape.batch}, got {mask.shape[0]}")
    binary = mask >= 0.5
    batch, _, pixel_frames, height, width = binary.shape
    spatial = torch.nn.functional.adaptive_max_pool2d(
        binary.float().movedim(2, 1).reshape(batch * pixel_frames, 1, height, width),
        (shape.height, shape.width),
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
        ).amax(dim=3)
        latent_mask = torch.cat((first, rest), dim=2)
    return latent_mask.to(torch.bool).permute(0, 2, 3, 4, 1).reshape(batch, -1)


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


def _normalize_stage2_input_cache_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized = dict(metadata)
    config = normalized.get("stage_1_config")
    if isinstance(config, dict):
        config = dict(config)
        # Older caches predate these explicit defaults. Fill them only for
        # validation so compatible Stage-2 latents can still be reused.
        config.setdefault("no_source_video_conditioning", False)
        config.setdefault("stage_1_ic_lora_strength", 1.0)
        normalized["stage_1_config"] = config
        normalized["stage_1_fingerprint"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
    return normalized


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
        metadata_for_compare = _normalize_stage2_input_cache_metadata(metadata)
        expected_for_compare = _normalize_stage2_input_cache_metadata(expected_metadata)
        mismatches = {
            key: (metadata_for_compare.get(key), value)
            for key, value in expected_for_compare.items()
            if metadata_for_compare.get(key) != value
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

def _model_response_metrics(model_input: torch.Tensor, x0: torch.Tensor, sigma: float) -> dict[str, object]:
    if sigma <= 0:
        raise ValueError(f"model-response diagnostics require positive sigma, got {sigma}")
    input_f = model_input.float()
    x0_f = x0.float()
    velocity = (input_f - x0_f) / sigma
    return {
        "input_norm": float(torch.linalg.vector_norm(input_f).cpu()),
        "x0_norm": float(torch.linalg.vector_norm(x0_f).cpu()),
        "velocity_norm": float(torch.linalg.vector_norm(velocity).cpu()),
        "input_finite": bool(torch.isfinite(input_f).all().cpu()),
        "x0_finite": bool(torch.isfinite(x0_f).all().cpu()),
        "velocity_finite": bool(torch.isfinite(velocity).all().cpu()),
    }



class Stage2PredictionRecorder:
    """Stream target-only branch predictions and summary metrics to disk."""

    def __init__(
        self,
        output_dir: str | Path,
        metadata: dict[str, object],
        routing_mask: torch.Tensor | None,
        reference_mask: torch.Tensor | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.routing_mask = routing_mask.detach().cpu() if routing_mask is not None else None
        self.reference_mask = reference_mask.detach().cpu() if reference_mask is not None else None
        self.manifest = {**metadata, "steps": []}
        if self.routing_mask is not None:
            save_file({"routing_mask": self.routing_mask.contiguous()}, self.output_dir / "routing_mask.safetensors")
        if self.reference_mask is not None:
            save_file(
                {"reference_mask": self.reference_mask.contiguous()},
                self.output_dir / "reference_mask.safetensors",
            )

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
        if step_index == 0:
            responses = {"routed": _model_response_metrics(shared_input, routed_x0, sigma)}
            if image_x0 is not None:
                responses["image"] = _model_response_metrics(shared_input, image_x0, sigma)
            if video_x0 is not None:
                responses["video"] = _model_response_metrics(shared_input, video_x0, sigma)
            row["first_model_response"] = responses
            row["shared_input_checksum"] = tensor_checksum(shared_input)
            row["routed_x0_checksum"] = tensor_checksum(routed_x0)
        if image_x0 is not None and video_x0 is not None:
            delta = video_x0 - image_x0
            row["branch_delta_rms"] = float(delta.float().square().mean().sqrt().cpu())
            if self.routing_mask is not None:
                mask = self.routing_mask.to(delta.device)
                row["branch_delta_rms_inside_mask"] = _weighted_rms(delta, mask)
                row["branch_delta_rms_outside_mask"] = _weighted_rms(delta, 1.0 - mask)
        self.manifest["steps"].append(row)
        (self.output_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def record_dual(  # noqa: PLR0913
        self,
        *,
        step_index: int,
        sigma: float,
        video_input: torch.Tensor,
        hybrid_input: torch.Tensor,
        video_x0: torch.Tensor,
        image_x0: torch.Tensor,
        next_video: torch.Tensor,
        next_hybrid: torch.Tensor,
        video_weight: torch.Tensor,
        kv_metrics: dict[str, object],
        reference_input: torch.Tensor | None = None,
        next_reference: torch.Tensor | None = None,
    ) -> None:
        tensors = {
            "video_input": video_input.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "hybrid_input": hybrid_input.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "video_x0": video_x0.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "image_x0": image_x0.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "next_video": next_video.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "next_hybrid": next_hybrid.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "video_weight": video_weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        }
        if reference_input is not None:
            tensors["reference_input"] = reference_input.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if next_reference is not None:
            tensors["next_reference"] = next_reference.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        filename = f"step_{step_index:03d}.safetensors"
        save_file(tensors, self.output_dir / filename, metadata={"step": str(step_index), "sigma": repr(sigma)})
        delta = video_x0 - image_x0
        row: dict[str, object] = {
            "step": step_index,
            "sigma": sigma,
            "file": filename,
            "video_input_checksum": tensor_checksum(video_input),
            "hybrid_input_checksum": tensor_checksum(hybrid_input),
            "next_video_checksum": tensor_checksum(next_video),
            "next_hybrid_checksum": tensor_checksum(next_hybrid),
            "reference_input_checksum": tensor_checksum(reference_input) if reference_input is not None else None,
            "next_reference_checksum": tensor_checksum(next_reference) if next_reference is not None else None,
            "branch_delta_rms": float(delta.float().square().mean().sqrt().cpu()),
            **kv_metrics,
        }
        if step_index == 0:
            row["first_model_response"] = {
                "video": _model_response_metrics(video_input, video_x0, sigma),
                "hybrid": _model_response_metrics(hybrid_input, image_x0, sigma),
            }
            row["video_x0_checksum"] = tensor_checksum(video_x0)
            row["hybrid_x0_checksum"] = tensor_checksum(image_x0)
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
    stage_2_noise_config: Stage2NoiseConfig | None = None,
) -> LatentState:
    if state.latent.shape[1] == target_token_count:
        return state
    latent = state.latent.clone()
    suffix = slice(target_token_count, None)
    noise = draw_or_replay_stage2_noise(
        bundle=stage_2_noise_config.noise_bundle if stage_2_noise_config is not None else None,
        role="source_video",
        shape=latent[:, suffix].shape,
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
    stage_2_noise_config: Stage2NoiseConfig | None = None,
    video_branch_runtime_lora: bool = True,
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
    stage_2_noise_config = stage_2_noise_config or Stage2NoiseConfig()

    image_state = create_noised_state(
        tools=video_tools,
        conditionings=image_conditionings,
        noiser=make_stage2_noiser(
            generator=target_generator,
            video_tools=video_tools,
            reference_latent=initial_video_latent,
            config=stage_2_noise_config,
        ),
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
        stage_2_noise_config=stage_2_noise_config,
    )
    shared_state = image_state
    base_audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=[],
        noiser=Stage2RoleNoiser(
            generator=audio_generator,
            bundle=stage_2_noise_config.noise_bundle,
            role="audio",
        ),
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
            lora_context = runtime_lora_scale(transformer, 1.0) if video_branch_runtime_lora else nullcontext()
            with lora_context:
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


def run_dual_trajectory_stage2(  # noqa: PLR0912,PLR0913,PLR0915
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
    video_edit_denoiser: Denoiser | None = None,
    noise_seed: int,
    routing_mask: torch.Tensor,
    dress_video_contribution: float,
    kv_controller: Stage2KVAttentionController,
    image_branch_ic_lora: bool = True,
    reference_tools: VideoLatentTools | None = None,
    initial_reference_latent: torch.Tensor | None = None,
    reference_denoiser: Denoiser | None = None,
    reference_noise_seed: int | None = None,
    reference_controller: Stage2ReferenceAttentionController | None = None,
    reference_edit_conditioning: str = "none",
    source_token_counts: tuple[int, ...] = (),
    recorder: Stage2PredictionRecorder | None = None,
    video_denoise_mask: torch.Tensor | None = None,
    stage_2_noise_config: Stage2NoiseConfig | None = None,
    video_branch_runtime_lora: bool = True,
) -> tuple[LatentState, LatentState]:
    """Run independent video-anchor and image-edit trajectories with optional KV guidance."""

    if not 0.0 <= dress_video_contribution <= 1.0:
        raise ValueError("dress_video_contribution must be in [0,1] for dual Stage-2 routing")
    if reference_edit_conditioning not in {"none", "video"}:
        raise ValueError("reference_edit_conditioning must be 'none' or 'video'")
    if reference_edit_conditioning == "video" and video_edit_denoiser is None:
        raise ValueError("video_edit_denoiser is required for a source-video-conditioned edit branch")
    target_count = video_tools.target_shape.token_count()
    if routing_mask.shape != (initial_video_latent.shape[0], target_count):
        raise ValueError(
            f"dual routing mask shape {tuple(routing_mask.shape)} does not match target tokens "
            f"({initial_video_latent.shape[0]},{target_count})"
        )
    noise_scale = float(sigmas[0].item())
    target_generator = torch.Generator(device=initial_video_latent.device).manual_seed(noise_seed)
    source_generator = torch.Generator(device=initial_video_latent.device).manual_seed(noise_seed + 1)
    audio_generator = torch.Generator(device=initial_audio_latent.device).manual_seed(noise_seed + 2)
    stage_2_noise_config = stage_2_noise_config or Stage2NoiseConfig()
    reference_values = (reference_tools, initial_reference_latent, reference_denoiser, reference_controller)
    has_any_reference_value = any(value is not None for value in reference_values)
    has_all_reference_values = all(value is not None for value in reference_values)
    if has_any_reference_value and not has_all_reference_values:
        raise ValueError("reference tools, latent, denoiser, and controller must be provided together")
    if reference_controller is not None and kv_controller.mode.value != "none":
        raise ValueError("reference appearance attention cannot be combined with video-anchor KV guidance")

    hybrid_state = create_noised_state(
        tools=video_tools,
        conditionings=image_conditionings,
        noiser=make_stage2_noiser(
            generator=target_generator,
            video_tools=video_tools,
            reference_latent=initial_video_latent,
            config=stage_2_noise_config,
        ),
        dtype=initial_video_latent.dtype,
        device=initial_video_latent.device,
        noise_scale=noise_scale,
        denoise_mask=video_denoise_mask,
        initial_latent=initial_video_latent,
    )
    video_state = state_with_conditionings(hybrid_state.clone(), video_conditionings, video_tools)
    video_state = _noise_appended_tokens(
        video_state,
        target_token_count=target_count,
        noise_scale=noise_scale,
        generator=source_generator,
    )
    if reference_edit_conditioning == "video":
        # Clone after source-token noising so anchor and edit branches start with
        # identical target and source tokens, then evolve independently.
        hybrid_state = video_state.clone()
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
    reference_state = None
    if reference_tools is not None and initial_reference_latent is not None:
        reference_generator = torch.Generator(device=initial_reference_latent.device).manual_seed(
            reference_noise_seed if reference_noise_seed is not None else noise_seed + 3
        )
        reference_state = create_noised_state(
            tools=reference_tools,
            conditionings=[],
            noiser=GaussianNoiser(reference_generator),
            dtype=initial_reference_latent.dtype,
            device=initial_reference_latent.device,
            noise_scale=noise_scale,
            initial_latent=initial_reference_latent,
        )
    soft_mask = routing_mask.to(device=hybrid_state.latent.device, dtype=hybrid_state.latent.dtype).unsqueeze(-1)
    video_weight = 1.0 - soft_mask * (1.0 - dress_video_contribution)
    stepper = EulerDiffusionStep()

    active_controller = reference_controller if reference_controller is not None else kv_controller
    with active_controller.patch_transformer(transformer):
        for step_index in tqdm(range(sigmas.numel() - 1), desc="Stage 2 dual-trajectory denoising"):
            if reference_controller is not None:
                reference_controller.begin_step(step_index)
            else:
                kv_controller.cache.begin_step(step_index)
            video_input = video_state.latent[:, :target_count].clone()
            hybrid_input = hybrid_state.latent[:, :target_count].clone()
            reference_input = reference_state.latent.clone() if reference_state is not None else None

            active_controller.phase = (
                "anchor"
                if reference_controller is not None
                and reference_controller.mode.value
                in {"source-qk-reference-v", "source-q-reference-kv"}
                else (None if reference_controller is not None else "capture")
            )
            video_lora_context = (
                runtime_lora_scale(transformer, 1.0) if video_branch_runtime_lora else nullcontext()
            )
            with video_lora_context:
                video_result, video_audio_result = video_denoiser(
                    transformer, video_state, video_audio_state, sigmas, step_index
                )
            if video_result is None or video_audio_result is None:
                raise RuntimeError("video anchor branch did not return video and audio predictions")

            next_reference = None
            if reference_controller is not None and reference_controller.enabled:
                assert reference_state is not None
                assert reference_denoiser is not None
                reference_controller.phase = "reference"
                with runtime_lora_scale(transformer, 1.0 if image_branch_ic_lora else 0.0):
                    reference_result, _ = reference_denoiser(
                        transformer, reference_state, None, sigmas, step_index
                    )
                if reference_result is None:
                    raise RuntimeError("appearance reference branch did not return a video prediction")
                reference_x0 = post_process_latent(
                    reference_result.denoised,
                    reference_state.denoise_mask,
                    reference_state.clean_latent,
                )
                next_reference = stepper.step(reference_state.latent, reference_x0, sigmas, step_index)

            active_controller.phase = "hybrid" if reference_controller is not None else "inject"
            active_edit_denoiser = (
                video_edit_denoiser if reference_edit_conditioning == "video" else image_denoiser
            )
            assert active_edit_denoiser is not None
            with runtime_lora_scale(transformer, 1.0 if image_branch_ic_lora else 0.0):
                image_result, image_audio_result = active_edit_denoiser(
                    transformer, hybrid_state, image_audio_state, sigmas, step_index
                )
            if image_result is None or image_audio_result is None:
                raise RuntimeError("image edit branch did not return video and audio predictions")

            video_x0 = video_result.denoised[:, :target_count]
            image_x0 = image_result.denoised[:, :target_count]
            video_x0 = post_process_latent(
                video_x0,
                video_state.denoise_mask[:, :target_count],
                video_state.clean_latent[:, :target_count],
            )
            image_x0 = post_process_latent(
                image_x0,
                hybrid_state.denoise_mask[:, :target_count],
                hybrid_state.clean_latent[:, :target_count],
            )
            next_video = stepper.step(video_input, video_x0, sigmas, step_index)
            next_edit = stepper.step(hybrid_input, image_x0, sigmas, step_index)
            next_hybrid = next_edit * (1.0 - video_weight) + next_video * video_weight
            reference_metrics = reference_controller.step_metrics() if reference_controller is not None else {}

            if recorder is not None:
                recorder.record_dual(
                    step_index=step_index,
                    sigma=float(sigmas[step_index].detach().cpu()),
                    video_input=video_input,
                    hybrid_input=hybrid_input,
                    video_x0=video_x0,
                    image_x0=image_x0,
                    next_video=next_video,
                    next_hybrid=next_hybrid,
                    video_weight=video_weight,
                    kv_metrics={
                        "kv_mode": kv_controller.mode.value,
                        "kv_strength": kv_controller.strength,
                        "kv_include_inside_mask": kv_controller.include_inside_mask,
                        "kv_anchor_region": kv_controller.anchor_region,
                        "kv_layers": sorted(kv_controller.active_layers),
                        "kv_cache_bytes": kv_controller.cache.bytes_stored,
                        "kv_capture_seconds": kv_controller.cache.capture_seconds,
                        "kv_load_seconds": kv_controller.cache.load_seconds,
                        "video_ic_lora_scale": 1.0,
                        "hybrid_ic_lora_scale": 1.0 if image_branch_ic_lora else 0.0,
                        "reference_ic_lora_scale": 1.0 if image_branch_ic_lora else 0.0,
                        "reference_edit_conditioning": reference_edit_conditioning,
                        "anchor_source_token_counts": list(source_token_counts),
                        "edit_source_token_counts": (
                            list(source_token_counts) if reference_edit_conditioning == "video" else []
                        ),
                        **reference_metrics,
                    },
                    reference_input=reference_input,
                    next_reference=next_reference,
                )
            if reference_controller is not None:
                logging.info(
                    "[Stage2 reference] step=%d mode=%s layers=%d cache=%.2f GiB capture=%.3fs load=%.3fs",
                    step_index,
                    reference_controller.mode.value,
                    len(reference_controller.active_layers),
                    reference_controller.cache.bytes_stored / (1024**3),
                    reference_controller.cache.capture_seconds,
                    reference_controller.cache.load_seconds,
                )
            if kv_controller.mode.value != "none":
                logging.info(
                    "[Stage2 KV] step=%d mode=%s layers=%d cache=%.2f GiB capture=%.3fs load=%.3fs",
                    step_index,
                    kv_controller.mode.value,
                    len(kv_controller.active_layers),
                    kv_controller.cache.bytes_stored / (1024**3),
                    kv_controller.cache.capture_seconds,
                    kv_controller.cache.load_seconds,
                )

            full_video_latent = video_state.latent.clone()
            full_video_latent[:, :target_count] = next_video
            video_state = replace(video_state, latent=full_video_latent)
            full_hybrid_latent = hybrid_state.latent.clone()
            full_hybrid_latent[:, :target_count] = next_hybrid
            hybrid_state = replace(hybrid_state, latent=full_hybrid_latent)
            if reference_state is not None and next_reference is not None:
                reference_state = replace(reference_state, latent=next_reference)

            image_audio_x0 = post_process_latent(
                image_audio_result.denoised, image_audio_state.denoise_mask, image_audio_state.clean_latent
            )
            video_audio_x0 = post_process_latent(
                video_audio_result.denoised, video_audio_state.denoise_mask, video_audio_state.clean_latent
            )
            image_audio_state = replace(
                image_audio_state,
                latent=stepper.step(image_audio_state.latent, image_audio_x0, sigmas, step_index),
            )
            video_audio_state = replace(
                video_audio_state,
                latent=stepper.step(video_audio_state.latent, video_audio_x0, sigmas, step_index),
            )
            active_controller.cache.clear()

    hybrid_target_state = replace(
        hybrid_state,
        latent=hybrid_state.latent[:, :target_count],
        clean_latent=hybrid_state.clean_latent[:, :target_count],
        denoise_mask=hybrid_state.denoise_mask[:, :target_count],
        positions=hybrid_state.positions[:, :, :target_count],
        attention_mask=None,
    )
    return video_tools.unpatchify(hybrid_target_state), audio_tools.unpatchify(video_audio_state)
