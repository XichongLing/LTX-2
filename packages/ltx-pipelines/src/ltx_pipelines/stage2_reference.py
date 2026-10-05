"""RoPE-neutral reference appearance attention for controlled Stage-2 sampling."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum

import torch

from ltx_pipelines.stage2_kv import Stage2KVCache, _call_attention, parse_stage2_kv_layers


class Stage2ReferenceAttentionMode(str, Enum):
    NONE = "none"
    BLEND = "blend"
    REPLACE = "replace"
    CONCAT = "concat"
    SOURCE_QK_REFERENCE_V = "source-qk-reference-v"
    SOURCE_Q_REFERENCE_KV = "source-q-reference-kv"


class Stage2ReferencePositionMode(str, Enum):
    PRE_ROPE = "pre-rope"
    POST_ROPE = "post-rope"
    Q_POST_K_PRE = "q-post-k-pre"


class _ReferencePreAttentionCallable:
    def __init__(
        self,
        *,
        original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        controller: "Stage2ReferenceAttentionController",
        layer_index: int,
    ) -> None:
        self.original = original
        self.controller = controller
        self.layer_index = layer_index

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: torch.nn.Module,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.controller.needs_pre_rope_features:
            self.controller.set_pre_rope(
                self.layer_index,
                attn_module.q_norm(q),
                attn_module.k_norm(k),
            )
        return self.original(q, k, attn_module, mask, pe, k_pe)


class _ReferenceAttentionCallable:
    def __init__(
        self,
        *,
        original: Callable[..., torch.Tensor],
        controller: "Stage2ReferenceAttentionController",
        layer_index: int,
    ) -> None:
        self.original = original
        self.controller = controller
        self.layer_index = layer_index

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controller.attend(
            layer_index=self.layer_index,
            original=self.original,
            q=q,
            k=k,
            v=v,
            heads=heads,
            mask=mask,
        )


class Stage2ReferenceAttentionController:
    """Capture masked pre-RoPE reference K/V and guide only masked target queries."""

    def __init__(
        self,
        *,
        target_token_count: int,
        target_hard_mask: torch.Tensor,
        reference_token_count: int,
        reference_hard_mask: torch.Tensor,
        mode: Stage2ReferenceAttentionMode | str,
        strength: float,
        layer_spec: str,
        cache: Stage2KVCache,
        position_mode: Stage2ReferencePositionMode | str = Stage2ReferencePositionMode.PRE_ROPE,
    ) -> None:
        self.mode = Stage2ReferenceAttentionMode(mode)
        self.position_mode = Stage2ReferencePositionMode(position_mode)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Stage-2 reference KV strength must be in [0,1], got {strength}")
        if self.mode in {
            Stage2ReferenceAttentionMode.REPLACE,
            Stage2ReferenceAttentionMode.SOURCE_QK_REFERENCE_V,
            Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV,
        } and strength != 1.0:
            raise ValueError(f"Stage-2 reference {self.mode.value} mode requires strength=1")
        if target_hard_mask.shape != (1, target_token_count):
            raise ValueError(
                f"target hard mask must have shape (1,{target_token_count}), got {tuple(target_hard_mask.shape)}"
            )
        if reference_hard_mask.shape != (1, reference_token_count):
            raise ValueError(
                "reference hard mask must have shape "
                f"(1,{reference_token_count}), got {tuple(reference_hard_mask.shape)}"
            )
        if self.mode is not Stage2ReferenceAttentionMode.NONE:
            if not target_hard_mask.any():
                raise ValueError("Stage-2 reference attention target mask selects no tokens")
            if not reference_hard_mask.any():
                raise ValueError("Stage-2 appearance reference mask selects no tokens")

        self.target_token_count = target_token_count
        self.target_hard_mask = target_hard_mask.to(dtype=torch.bool)
        self.reference_token_count = reference_token_count
        self.reference_hard_mask = reference_hard_mask.to(dtype=torch.bool)
        self.strength = strength
        self.layer_spec = layer_spec
        self.cache = cache
        self.active_layers: set[int] = set()
        self.phase: str | None = None
        self._pre_rope: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._unmasked_attention: dict[int, Callable[..., torch.Tensor]] = {}
        self._masked_attention: dict[int, Callable[..., torch.Tensor]] = {}
        self._inside_delta_sum = 0.0
        self._inside_delta_count = 0
        self._anchor_q: dict[int, torch.Tensor] = {}
        self._anchor_k: dict[int, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return self.mode is not Stage2ReferenceAttentionMode.NONE and self.strength > 0.0

    @property
    def needs_pre_rope_features(self) -> bool:
        return self.enabled and (
            (
                self.position_mode is Stage2ReferencePositionMode.PRE_ROPE
                and self.phase in {"anchor", "reference", "hybrid"}
            )
            or (
                self.position_mode is Stage2ReferencePositionMode.Q_POST_K_PRE
                and self.phase == "reference"
            )
        )

    @property
    def selected_reference_tokens(self) -> int:
        return int(self.reference_hard_mask.sum().item())

    @property
    def selected_target_tokens(self) -> int:
        return int(self.target_hard_mask.sum().item())

    def begin_step(self, step_index: int) -> None:
        self.cache.begin_step(step_index)
        self._pre_rope.clear()
        self._anchor_q.clear()
        self._anchor_k.clear()
        self._inside_delta_sum = 0.0
        self._inside_delta_count = 0

    def set_pre_rope(self, layer_index: int, q: torch.Tensor, k: torch.Tensor) -> None:
        self._pre_rope[layer_index] = (q, k)

    def step_metrics(self) -> dict[str, object]:
        mean_delta = self._inside_delta_sum / max(self._inside_delta_count, 1)
        return {
            "reference_attention_mode": self.mode.value,
            "reference_position_mode": self.position_mode.value,
            "reference_kv_strength": self.strength,
            "reference_kv_layers": sorted(self.active_layers),
            "reference_selected_tokens": self.selected_reference_tokens,
            "target_selected_tokens": self.selected_target_tokens,
            "reference_kv_cache_bytes": self.cache.bytes_stored,
            "reference_kv_capture_seconds": self.cache.capture_seconds,
            "reference_kv_load_seconds": self.cache.load_seconds,
            "reference_attention_inside_delta_rms_mean": mean_delta,
            "reference_attention_outside_delta_rms": 0.0,
        }

    @contextmanager
    def patch_transformer(self, transformer: torch.nn.Module) -> Iterator[None]:
        velocity_model = getattr(transformer, "velocity_model", None)
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if blocks is None:
            raise TypeError("Stage-2 reference attention requires a normal single-GPU LTX transformer")
        self.active_layers = parse_stage2_kv_layers(self.layer_spec, len(blocks))
        originals: list[tuple[object, object, object, object]] = []
        try:
            for layer_index, block in enumerate(blocks):
                if layer_index not in self.active_layers:
                    continue
                attention = block.attn1
                originals.append(
                    (
                        attention,
                        attention.preattention_function,
                        attention.attention_function,
                        attention.masked_attention_function,
                    )
                )
                self._unmasked_attention[layer_index] = attention.attention_function
                self._masked_attention[layer_index] = attention.masked_attention_function
                attention.preattention_function = _ReferencePreAttentionCallable(
                    original=attention.preattention_function,
                    controller=self,
                    layer_index=layer_index,
                )
                attention.attention_function = _ReferenceAttentionCallable(
                    original=attention.attention_function,
                    controller=self,
                    layer_index=layer_index,
                )
                attention.masked_attention_function = _ReferenceAttentionCallable(
                    original=attention.masked_attention_function,
                    controller=self,
                    layer_index=layer_index,
                )
            yield
        finally:
            self.phase = None
            self._pre_rope.clear()
            self._anchor_q.clear()
            self._anchor_k.clear()
            self._unmasked_attention.clear()
            self._masked_attention.clear()
            self.cache.clear()
            for attention, original_pre, original_unmasked, original_masked in originals:
                attention.preattention_function = original_pre
                attention.attention_function = original_unmasked
                attention.masked_attention_function = original_masked

    def attend(  # noqa: PLR0912, PLR0915
        self,
        *,
        layer_index: int,
        original: Callable[..., torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enabled or self.phase is None:
            return _call_attention(original, q, k, v, heads, mask)
        if self.position_mode is Stage2ReferencePositionMode.PRE_ROPE or (
            self.position_mode is Stage2ReferencePositionMode.Q_POST_K_PRE and self.phase == "reference"
        ):
            try:
                guide_q, guide_k = self._pre_rope.pop(layer_index)
            except KeyError as exc:
                raise RuntimeError(f"missing pre-RoPE features for Stage-2 reference layer {layer_index}") from exc
        else:
            guide_q, guide_k = q, k

        if self.phase == "anchor":
            if self.mode not in {
                Stage2ReferenceAttentionMode.SOURCE_QK_REFERENCE_V,
                Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV,
            }:
                return _call_attention(original, q, k, v, heads, mask)
            if guide_q.shape[0] != 1 or guide_q.shape[1] < self.target_token_count:
                raise ValueError(
                    "source-Q/K reference-V mode requires one anchor batch containing at least "
                    f"{self.target_token_count} target tokens"
                )
            inside = self.target_hard_mask[0].to(guide_q.device).nonzero(as_tuple=False).flatten()
            self._anchor_q[layer_index] = guide_q[:, : self.target_token_count].index_select(1, inside).to(
                device="cpu", dtype=torch.bfloat16
            )
            if self.mode is Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV:
                return _call_attention(original, q, k, v, heads, mask)
            if self.reference_token_count > self.target_token_count:
                raise ValueError(
                    "source-Q/K reference-V mode requires the reference token grid not to exceed "
                    f"the target grid ({self.reference_token_count} > {self.target_token_count})"
                )
            reference_positions = self.reference_hard_mask[0].to(guide_k.device).nonzero(as_tuple=False).flatten()
            # Reference tokens use the same temporal-major flattened grid as target
            # tokens. A one-frame reference pairs with the anchor's first latent
            # frame; a full-length repeated reference pairs all frames.
            self._anchor_k[layer_index] = guide_k[:, : self.reference_token_count].index_select(
                1, reference_positions
            ).to(device="cpu", dtype=torch.bfloat16)
            return _call_attention(original, q, k, v, heads, mask)

        if self.phase == "reference":
            if guide_k.shape[1] != self.reference_token_count or v.shape[1] != self.reference_token_count:
                raise ValueError(
                    "reference attention token count mismatch: "
                    f"guide_k={guide_k.shape[1]}, v={v.shape[1]}, expected={self.reference_token_count}"
                )
            reference_indices = self.reference_hard_mask[0].to(guide_k.device).nonzero(as_tuple=False).flatten()
            self.cache.put(
                layer_index,
                guide_k.index_select(1, reference_indices),
                v.index_select(1, reference_indices),
            )
            return _call_attention(original, q, k, v, heads, mask)

        if self.phase != "hybrid":
            raise RuntimeError(f"unknown Stage-2 reference attention phase: {self.phase}")
        if q.shape[0] != 1 or q.shape[1] < self.target_token_count:
            raise ValueError(
                "Stage-2 reference attention requires one hybrid batch containing at least "
                f"{self.target_token_count} target tokens"
            )

        base = _call_attention(original, q, k, v, heads, mask)
        inside = self.target_hard_mask[0].to(q.device).nonzero(as_tuple=False).flatten()
        reference_k, reference_v = self.cache.consume(layer_index, device=q.device, dtype=guide_q.dtype)
        base_inside = base.index_select(1, inside)
        if self.mode is Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV:
            try:
                anchor_q = self._anchor_q.pop(layer_index).to(device=q.device, dtype=guide_q.dtype)
            except KeyError as exc:
                raise RuntimeError(f"missing anchor Q for Stage-2 reference layer {layer_index}") from exc
            guided = _call_attention(
                self._unmasked_attention.get(layer_index, original),
                anchor_q,
                reference_k,
                reference_v.to(dtype=v.dtype),
                heads,
                None,
            )
        elif self.mode is Stage2ReferenceAttentionMode.SOURCE_QK_REFERENCE_V:
            try:
                anchor_q = self._anchor_q.pop(layer_index).to(device=q.device, dtype=guide_q.dtype)
                anchor_k = self._anchor_k.pop(layer_index).to(device=q.device, dtype=guide_q.dtype)
            except KeyError as exc:
                raise RuntimeError(f"missing anchor Q/K for Stage-2 reference layer {layer_index}") from exc
            if anchor_k.shape[1] != reference_v.shape[1]:
                raise ValueError(
                    "source K/reference V token mismatch: "
                    f"source_k={anchor_k.shape[1]}, reference_v={reference_v.shape[1]}"
                )
            guided = _call_attention(
                self._unmasked_attention.get(layer_index, original),
                anchor_q,
                anchor_k,
                reference_v.to(dtype=v.dtype),
                heads,
                None,
            )
        elif self.mode is Stage2ReferenceAttentionMode.CONCAT:
            native_mask = _select_query_rows(mask, inside)
            if native_mask is None:
                native_mask = torch.zeros(
                    (inside.numel(), guide_k.shape[1]),
                    device=q.device,
                    dtype=guide_q.dtype,
                )
            reference_mask = torch.full(
                (*native_mask.shape[:-1], reference_k.shape[1]),
                math.log(self.strength),
                device=native_mask.device,
                dtype=native_mask.dtype,
            )
            guided = _call_attention(
                self._masked_attention[layer_index],
                guide_q.index_select(1, inside),
                torch.cat((guide_k, reference_k), dim=1),
                torch.cat((v, reference_v.to(dtype=v.dtype)), dim=1),
                heads,
                torch.cat((native_mask, reference_mask), dim=-1),
            )
        else:
            reference_out = _call_attention(
                self._unmasked_attention.get(layer_index, original),
                guide_q.index_select(1, inside),
                reference_k,
                reference_v.to(dtype=v.dtype),
                heads,
                None,
            )
            effective_strength = 1.0 if self.mode is Stage2ReferenceAttentionMode.REPLACE else self.strength
            guided = torch.lerp(base_inside, reference_out.to(base_inside.dtype), effective_strength)
        delta = guided.float() - base_inside.float()
        self._inside_delta_sum += float(delta.square().mean().sqrt().detach().cpu())
        self._inside_delta_count += 1
        output = base.clone()
        output[:, inside] = guided
        return output


def _select_query_rows(mask: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.index_select(0, indices)
    if mask.ndim == 3:
        return mask.index_select(1, indices)
    if mask.ndim == 4:
        return mask.index_select(2, indices)
    raise ValueError(f"unsupported Stage-2 reference attention mask rank: {mask.ndim}")
