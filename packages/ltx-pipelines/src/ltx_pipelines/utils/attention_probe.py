from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ltx_core.types import LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.helpers import post_process_latent


def parse_index_set(spec: str | None) -> set[int] | None:
    """Parse a comma-separated index/range spec. None means all indices."""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in {"", "all", "*"}:
        return None

    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending range in index spec: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


@dataclass(frozen=True)
class AttentionProbeConfig:
    output_path: Path
    layers: set[int] | None = None
    steps: set[int] | None = None
    heads: str = "mean"
    query_chunk_size: int = 128
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StageLayout:
    stage: int
    target_token_count: int
    total_token_count: int
    reference_token_start: int
    reference_token_count: int
    first_frame_token_count: int
    tokens_per_frame: int
    target_latent_frames: int


class AttentionProbe:
    """Aggregate attention/value metrics for video self-attention in LTX blocks."""

    def __init__(self, config: AttentionProbeConfig) -> None:
        if config.heads not in {"mean", "all"}:
            raise ValueError(f"attention probe heads must be 'mean' or 'all', got {config.heads!r}")
        if config.query_chunk_size <= 0:
            raise ValueError("attention probe query_chunk_size must be positive")

        self.config = config
        self.output_path = Path(config.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_step: int | None = None
        self._current_sigma: float | None = None
        self._layout: _StageLayout | None = None
        self._patched_attention_ids: set[int] = set()

        self.output_path.write_text("", encoding="utf-8")
        self._write_json(
            {
                "event": "probe_start",
                "time": time.time(),
                "layers": None if config.layers is None else sorted(config.layers),
                "steps": None if config.steps is None else sorted(config.steps),
                "heads": config.heads,
                "query_chunk_size": config.query_chunk_size,
                "metadata": config.metadata,
            }
        )

    def make_loop(
        self,
        *,
        stage: int,
        width: int,
        height: int,
        frames: int,
        fps: float,
    ):
        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
        target_shape = VideoLatentShape.from_pixel_shape(pixel_shape)

        def loop(
            sigmas: torch.Tensor,
            video_state: LatentState | None,
            audio_state: LatentState | None,
            stepper,
            transformer,
            denoiser,
        ) -> tuple[LatentState | None, LatentState | None]:
            if video_state is not None:
                self._configure_stage(stage=stage, target_shape=target_shape, video_state=video_state)
                self.patch_transformer(transformer)

            try:
                for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
                    self._current_step = step_idx
                    self._current_sigma = float(sigmas[step_idx].detach().cpu())
                    denoised_video, denoised_audio = denoiser(
                        transformer, video_state, audio_state, sigmas, step_idx
                    )

                    video_state = self._step_state(video_state, denoised_video, stepper, sigmas, step_idx)
                    audio_state = self._step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)
            finally:
                self._current_step = None
                self._current_sigma = None

            return video_state, audio_state

        return loop

    def patch_transformer(self, transformer: torch.nn.Module) -> None:
        velocity_model = getattr(transformer, "velocity_model", None)
        if velocity_model is None:
            return
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if blocks is None:
            return

        for layer_idx, block in enumerate(blocks):
            if self.config.layers is not None and layer_idx not in self.config.layers:
                continue
            attention = getattr(block, "attn1", None)
            if attention is None:
                continue
            if id(attention) in self._patched_attention_ids:
                continue
            attention.attention_function = _ProbedAttentionCallable(
                original=attention.attention_function,
                probe=self,
                layer_idx=layer_idx,
            )
            self._patched_attention_ids.add(id(attention))

    def should_probe(self, layer_idx: int) -> bool:
        if self._layout is None or self._current_step is None:
            return False
        if self.config.layers is not None and layer_idx not in self.config.layers:
            return False
        if self.config.steps is not None and self._current_step not in self.config.steps:
            return False
        return True

    @torch.no_grad()
    def collect(self, *, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask) -> None:
        layout = self._layout
        if layout is None:
            return
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            return

        batch_size, seq_len, inner_dim = q.shape
        if layout.target_token_count <= 0 or layout.target_token_count > seq_len:
            return

        dim_head = inner_dim // heads
        scale = 1.0 / math.sqrt(dim_head)
        target_count = layout.target_token_count
        first_count = min(layout.first_frame_token_count, target_count)
        future_start = min(first_count, target_count)
        ref_start = min(layout.reference_token_start, seq_len)
        ref_count = max(0, seq_len - ref_start)

        q_heads = q.view(batch_size, seq_len, heads, dim_head).transpose(1, 2)
        k_heads = k.view(batch_size, seq_len, heads, dim_head).transpose(1, 2).float()
        v_heads = v.view(batch_size, seq_len, heads, dim_head).transpose(1, 2).float()

        device = q.device
        target_ref_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        target_first_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        future_first_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        ref_value_norm_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        first_value_norm_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        total_value_norm_sum = torch.zeros(heads, device=device, dtype=torch.float64)
        frame_ref_sums = torch.zeros(heads, layout.target_latent_frames, device=device, dtype=torch.float64)
        frame_denoms = torch.zeros(layout.target_latent_frames, device=device, dtype=torch.float64)

        target_denom = 0
        future_denom = 0
        chunk_size = self.config.query_chunk_size

        for q0 in range(0, target_count, chunk_size):
            q1 = min(q0 + chunk_size, target_count)
            q_chunk = q_heads[:, :, q0:q1, :].float()
            scores = torch.matmul(q_chunk, k_heads.transpose(-2, -1)) * scale
            bias = _mask_chunk(mask, q0, q1, scores.dtype)
            if bias is not None:
                scores = scores + bias

            probs = torch.softmax(scores, dim=-1)
            q_count = q1 - q0
            target_denom += batch_size * q_count

            if ref_count > 0:
                ref_probs = probs[..., ref_start:]
                target_ref_sum += ref_probs.sum(dim=(0, 2, 3), dtype=torch.float64)
                ref_out = torch.matmul(ref_probs, v_heads[:, :, ref_start:, :])
                ref_value_norm_sum += ref_out.norm(dim=-1).sum(dim=(0, 2), dtype=torch.float64)

            if first_count > 0:
                first_probs = probs[..., :first_count]
                target_first_sum += first_probs.sum(dim=(0, 2, 3), dtype=torch.float64)
                first_out = torch.matmul(first_probs, v_heads[:, :, :first_count, :])
                first_value_norm_sum += first_out.norm(dim=-1).sum(dim=(0, 2), dtype=torch.float64)

                future_q0 = max(q0, future_start)
                if future_q0 < q1:
                    local0 = future_q0 - q0
                    future_first_sum += probs[:, :, local0:, :first_count].sum(
                        dim=(0, 2, 3), dtype=torch.float64
                    )
                    future_denom += batch_size * (q1 - future_q0)

            total_out = torch.matmul(probs, v_heads)
            total_value_norm_sum += total_out.norm(dim=-1).sum(dim=(0, 2), dtype=torch.float64)

            if ref_count > 0 and layout.tokens_per_frame > 0:
                for frame_idx in range(layout.target_latent_frames):
                    f0 = frame_idx * layout.tokens_per_frame
                    f1 = min(f0 + layout.tokens_per_frame, target_count)
                    overlap0 = max(q0, f0)
                    overlap1 = min(q1, f1)
                    if overlap0 >= overlap1:
                        continue
                    local0 = overlap0 - q0
                    local1 = overlap1 - q0
                    frame_ref_sums[:, frame_idx] += probs[:, :, local0:local1, ref_start:].sum(
                        dim=(0, 2, 3), dtype=torch.float64
                    )
                    frame_denoms[frame_idx] += batch_size * (overlap1 - overlap0)

            del scores, probs, q_chunk

        record_base = {
            "event": "attention_probe",
            "stage": layout.stage,
            "step": self._current_step,
            "sigma": self._current_sigma,
            "layer": layer_idx,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_heads": heads,
            "target_token_count": target_count,
            "reference_token_start": ref_start,
            "reference_token_count": ref_count,
            "first_frame_token_count": first_count,
            "tokens_per_frame": layout.tokens_per_frame,
            "target_latent_frames": layout.target_latent_frames,
        }

        target_ref_mass = _safe_div_tensor(target_ref_sum, target_denom) if ref_count > 0 else None
        target_first_mass = _safe_div_tensor(target_first_sum, target_denom) if first_count > 0 else None
        future_first_mass = _safe_div_tensor(future_first_sum, future_denom) if future_denom > 0 else None
        ref_value_ratio = _safe_div_tensor(ref_value_norm_sum, total_value_norm_sum) if ref_count > 0 else None
        first_value_ratio = _safe_div_tensor(first_value_norm_sum, total_value_norm_sum) if first_count > 0 else None

        frame_ref_mass = None
        if ref_count > 0 and bool((frame_denoms > 0).any()):
            frame_ref_mass = frame_ref_sums / frame_denoms.clamp_min(1).unsqueeze(0)

        if self.config.heads == "all":
            for head_idx in range(heads):
                record = dict(record_base)
                record.update(
                    {
                        "head": head_idx,
                        "target_to_ref_video_mass": _tensor_item(target_ref_mass, head_idx),
                        "target_to_first_frame_mass": _tensor_item(target_first_mass, head_idx),
                        "future_target_to_first_frame_mass": _tensor_item(future_first_mass, head_idx),
                        "ref_video_value_contribution_ratio": _tensor_item(ref_value_ratio, head_idx),
                        "first_frame_value_contribution_ratio": _tensor_item(first_value_ratio, head_idx),
                        "target_frame_to_ref_video_mass": _tensor_list(frame_ref_mass, head_idx),
                    }
                )
                self._write_json(record)
        else:
            record = dict(record_base)
            record.update(
                {
                    "head": "mean",
                    "target_to_ref_video_mass": _tensor_mean(target_ref_mass),
                    "target_to_first_frame_mass": _tensor_mean(target_first_mass),
                    "future_target_to_first_frame_mass": _tensor_mean(future_first_mass),
                    "ref_video_value_contribution_ratio": _tensor_mean(ref_value_ratio),
                    "first_frame_value_contribution_ratio": _tensor_mean(first_value_ratio),
                    "target_frame_to_ref_video_mass": _tensor_mean_list(frame_ref_mass),
                }
            )
            self._write_json(record)

    def _configure_stage(
        self,
        *,
        stage: int,
        target_shape: VideoLatentShape,
        video_state: LatentState,
    ) -> None:
        target_count = target_shape.token_count()
        total_count = video_state.latent.shape[1]
        tokens_per_frame = target_shape.height * target_shape.width
        layout = _StageLayout(
            stage=stage,
            target_token_count=target_count,
            total_token_count=total_count,
            reference_token_start=min(target_count, total_count),
            reference_token_count=max(0, total_count - target_count),
            first_frame_token_count=tokens_per_frame,
            tokens_per_frame=tokens_per_frame,
            target_latent_frames=target_shape.frames,
        )
        self._layout = layout
        self._write_json(
            {
                "event": "stage_start",
                "stage": stage,
                "target_token_count": layout.target_token_count,
                "total_token_count": layout.total_token_count,
                "reference_token_start": layout.reference_token_start,
                "reference_token_count": layout.reference_token_count,
                "first_frame_token_count": layout.first_frame_token_count,
                "tokens_per_frame": layout.tokens_per_frame,
                "target_latent_frames": layout.target_latent_frames,
            }
        )

    @staticmethod
    def _step_state(
        state: LatentState | None,
        denoised: torch.Tensor | None,
        stepper,
        sigmas: torch.Tensor,
        step_idx: int,
    ) -> LatentState | None:
        if state is None or denoised is None:
            return state
        denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        from dataclasses import replace

        return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))

    def _write_json(self, record: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class _ProbedAttentionCallable:
    def __init__(self, *, original, probe: AttentionProbe, layer_idx: int) -> None:
        self.original = original
        self.probe = probe
        self.layer_idx = layer_idx

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.probe.should_probe(self.layer_idx):
            self.probe.collect(layer_idx=self.layer_idx, q=q, k=k, v=v, heads=heads, mask=mask)
        return self.original(q, k, v, heads, mask)


def _mask_chunk(mask: torch.Tensor | None, q0: int, q1: int, dtype: torch.dtype) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[q0:q1, :].to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        return mask[:, q0:q1, :].to(dtype=dtype).unsqueeze(1)
    if mask.ndim == 4:
        return mask[:, :, q0:q1, :].to(dtype=dtype)
    raise ValueError(f"Unsupported attention mask rank for probing: {mask.ndim}")


def _safe_div_tensor(numerator: torch.Tensor, denominator: int | torch.Tensor) -> torch.Tensor:
    if isinstance(denominator, int):
        if denominator <= 0:
            return torch.zeros_like(numerator)
        return numerator / float(denominator)
    return numerator / denominator.clamp_min(1e-12)


def _tensor_item(value: torch.Tensor | None, index: int) -> float | None:
    if value is None:
        return None
    return float(value[index].detach().cpu())


def _tensor_mean(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.mean().detach().cpu())


def _tensor_list(value: torch.Tensor | None, index: int) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value[index].detach().cpu().tolist()]


def _tensor_mean_list(value: torch.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value.mean(dim=0).detach().cpu().tolist()]
