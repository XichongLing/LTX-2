"""Stage-2 target-token KV capture and dress-query attention guidance."""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class Stage2KVMode(str, Enum):
    NONE = "none"
    PARTITION = "partition"
    APPEND = "append"


class Stage2KVCache:
    """Current-step-only compact KV storage with CPU and disk backends."""

    def __init__(self, *, backend: str = "cpu", cache_dir: str | Path | None = None) -> None:
        if backend not in {"cpu", "disk"}:
            raise ValueError(f"unsupported Stage-2 KV cache backend: {backend}")
        if backend == "disk" and cache_dir is None:
            raise ValueError("cache_dir is required for the disk Stage-2 KV cache backend")
        self.backend = backend
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self._step_index: int | None = None
        self._memory: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.bytes_stored = 0
        self.capture_seconds = 0.0
        self.load_seconds = 0.0

    @property
    def step_dir(self) -> Path | None:
        if self.cache_dir is None or self._step_index is None:
            return None
        return self.cache_dir / f"step_{self._step_index:03d}"

    def begin_step(self, step_index: int) -> None:
        self.clear()
        self._step_index = step_index
        self.bytes_stored = 0
        self.capture_seconds = 0.0
        self.load_seconds = 0.0
        if self.backend == "disk":
            assert self.step_dir is not None
            self.step_dir.mkdir(parents=True, exist_ok=True)

    def put(self, layer_index: int, k: torch.Tensor, v: torch.Tensor) -> None:
        start = time.perf_counter()
        tensors = {
            "k": k.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "v": v.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        }
        self.bytes_stored += sum(t.numel() * t.element_size() for t in tensors.values())
        if self.backend == "cpu":
            self._memory[layer_index] = (tensors["k"], tensors["v"])
        else:
            assert self.step_dir is not None
            save_file(tensors, self.step_dir / f"layer_{layer_index:03d}.safetensors")
        self.capture_seconds += time.perf_counter() - start

    def consume(
        self, layer_index: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = time.perf_counter()
        if self.backend == "cpu":
            try:
                k_cpu, v_cpu = self._memory.pop(layer_index)
            except KeyError as exc:
                raise KeyError(f"missing Stage-2 KV cache for layer {layer_index}") from exc
        else:
            assert self.step_dir is not None
            path = self.step_dir / f"layer_{layer_index:03d}.safetensors"
            if not path.is_file():
                raise KeyError(f"missing Stage-2 KV cache file for layer {layer_index}: {path}")
            tensors = load_file(path, device="cpu")
            k_cpu, v_cpu = tensors["k"], tensors["v"]
            path.unlink()
        k = k_cpu.to(device=device, dtype=dtype)
        v = v_cpu.to(device=device, dtype=dtype)
        self.load_seconds += time.perf_counter() - start
        return k, v

    def clear(self) -> None:
        self._memory.clear()
        step_dir = self.step_dir
        if step_dir is not None and step_dir.exists():
            shutil.rmtree(step_dir)
        self._step_index = None


def parse_stage2_kv_layers(spec: str, num_layers: int) -> set[int]:
    if spec.strip().lower() in {"all", "*", ""}:
        return set(range(num_layers))
    layers: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid descending Stage-2 KV layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"Stage-2 KV layers out of range [0,{num_layers - 1}]: {invalid}")
    return layers


def _call_attention(
    original: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return original(q, k, v, heads)
    return original(q, k, v, heads, mask)


def _select_query_rows(mask: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.index_select(0, indices)
    if mask.ndim == 3:
        return mask.index_select(1, indices)
    if mask.ndim == 4:
        return mask.index_select(2, indices)
    raise ValueError(f"unsupported Stage-2 attention mask rank: {mask.ndim}")


def _select_key_columns(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return mask.index_select(mask.ndim - 1, indices)


class _ControlledAttentionCallable:
    def __init__(
        self,
        *,
        original: Callable[..., torch.Tensor],
        controller: "Stage2KVAttentionController",
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


class Stage2KVAttentionController:
    """Capture anchor K/V and modify only dress queries in the edit pass."""

    def __init__(
        self,
        *,
        target_token_count: int,
        hard_mask: torch.Tensor,
        mode: Stage2KVMode | str,
        strength: float,
        layer_spec: str,
        cache: Stage2KVCache,
        include_inside_mask: bool = False,
        anchor_region: str = "outside",
    ) -> None:
        self.target_token_count = target_token_count
        self.mode = Stage2KVMode(mode)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Stage-2 KV strength must be in [0,1], got {strength}")
        self.strength = strength
        if hard_mask.shape != (1, target_token_count):
            raise ValueError(
                f"hard Stage-2 KV mask must have shape (1,{target_token_count}), got {tuple(hard_mask.shape)}"
            )
        self.hard_mask = hard_mask.to(dtype=torch.bool)
        self.layer_spec = layer_spec
        self.cache = cache
        if include_inside_mask:
            anchor_region = "all"
        if anchor_region not in {"outside", "inside", "all"}:
            raise ValueError(f"Stage-2 KV anchor region must be outside, inside, or all; got {anchor_region!r}")
        self.anchor_region = anchor_region
        self.include_inside_mask = anchor_region == "all"
        self.active_layers: set[int] = set()
        self.phase: str | None = None

    @contextmanager
    def patch_transformer(self, transformer: torch.nn.Module) -> Iterator[None]:
        velocity_model = getattr(transformer, "velocity_model", None)
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if blocks is None:
            raise TypeError("Stage-2 KV routing requires a normal single-GPU LTX transformer")
        self.active_layers = parse_stage2_kv_layers(self.layer_spec, len(blocks))
        originals: list[tuple[object, object, object]] = []
        try:
            for layer_index, block in enumerate(blocks):
                if layer_index not in self.active_layers:
                    continue
                attention = block.attn1
                originals.append((attention, attention.attention_function, attention.masked_attention_function))
                attention.attention_function = _ControlledAttentionCallable(
                    original=attention.attention_function, controller=self, layer_index=layer_index
                )
                attention.masked_attention_function = _ControlledAttentionCallable(
                    original=attention.masked_attention_function, controller=self, layer_index=layer_index
                )
            yield
        finally:
            self.phase = None
            self.cache.clear()
            for attention, original_unmasked, original_masked in originals:
                attention.attention_function = original_unmasked
                attention.masked_attention_function = original_masked

    def attend(  # noqa: PLR0912
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
        if self.mode is Stage2KVMode.NONE or self.strength == 0.0 or self.phase is None:
            return _call_attention(original, q, k, v, heads, mask)
        if self.phase == "capture":
            if k.shape[1] < self.target_token_count or v.shape[1] < self.target_token_count:
                raise ValueError("anchor self-attention has fewer tokens than the Stage-2 target")
            if self.anchor_region == "all":
                anchor_indices = torch.arange(self.target_token_count, device=k.device)
            elif self.anchor_region == "inside":
                anchor_indices = self.hard_mask[0].to(device=k.device).nonzero(as_tuple=False).flatten()
            else:
                anchor_indices = (~self.hard_mask[0]).to(device=k.device).nonzero(as_tuple=False).flatten()
            self.cache.put(
                layer_index,
                k[:, : self.target_token_count].index_select(1, anchor_indices),
                v[:, : self.target_token_count].index_select(1, anchor_indices),
            )
            return _call_attention(original, q, k, v, heads, mask)
        if self.phase != "inject":
            raise RuntimeError(f"unknown Stage-2 KV attention phase: {self.phase}")
        if q.shape[0] != 1 or q.shape[1] != self.target_token_count:
            raise ValueError("Stage-2 KV edit attention currently requires one target-only batch")

        inside = self.hard_mask[0].to(device=q.device).nonzero(as_tuple=False).flatten()
        outside = (~self.hard_mask[0]).to(device=q.device).nonzero(as_tuple=False).flatten()
        if inside.numel() == 0:
            self.cache.consume(layer_index, device=q.device, dtype=k.dtype)
            return _call_attention(original, q, k, v, heads, mask)

        out = torch.empty_like(q)
        if outside.numel() > 0:
            q_out = q.index_select(1, outside)
            out[:, outside] = _call_attention(
                original, q_out, k, v, heads, _select_query_rows(mask, outside)
            )

        anchor_k, anchor_v = self.cache.consume(layer_index, device=k.device, dtype=k.dtype)
        if self.anchor_region == "all":
            expected_anchor_tokens = self.target_token_count
        elif self.anchor_region == "inside":
            expected_anchor_tokens = inside.numel()
        else:
            expected_anchor_tokens = outside.numel()
        if anchor_k.shape[0] != k.shape[0] or anchor_k.shape[1] != expected_anchor_tokens:
            raise ValueError(
                f"cached anchor shape {tuple(anchor_k.shape)} is incompatible with edit K/V {tuple(k.shape)}"
            )
        q_in = q.index_select(1, inside)
        inside_mask = _select_query_rows(mask, inside)
        if self.mode is Stage2KVMode.PARTITION:
            mixed_k = k.clone()
            mixed_v = v.clone()
            if self.anchor_region == "all":
                mixed_k[:, : self.target_token_count] = torch.lerp(
                    mixed_k[:, : self.target_token_count], anchor_k, self.strength
                )
                mixed_v[:, : self.target_token_count] = torch.lerp(
                    mixed_v[:, : self.target_token_count], anchor_v, self.strength
                )
            elif self.anchor_region == "inside":
                mixed_k[:, inside] = torch.lerp(mixed_k[:, inside], anchor_k, self.strength)
                mixed_v[:, inside] = torch.lerp(mixed_v[:, inside], anchor_v, self.strength)
            else:
                mixed_k[:, outside] = torch.lerp(mixed_k[:, outside], anchor_k, self.strength)
                mixed_v[:, outside] = torch.lerp(mixed_v[:, outside], anchor_v, self.strength)
            guided = _call_attention(original, q_in, mixed_k, mixed_v, heads, inside_mask)
        else:
            appended_k = torch.cat((k, anchor_k), dim=1)
            appended_v = torch.cat((v, anchor_v), dim=1)
            if inside_mask is None and self.strength == 1.0:
                appended_mask = None
            else:
                if inside_mask is None:
                    inside_mask = torch.zeros(
                        (inside.numel(), self.target_token_count), device=q.device, dtype=q.dtype
                    )
                if self.anchor_region == "all":
                    anchor_columns = torch.arange(self.target_token_count, device=q.device)
                elif self.anchor_region == "inside":
                    anchor_columns = inside
                else:
                    anchor_columns = outside
                anchor_mask = _select_key_columns(inside_mask, anchor_columns)
                anchor_mask = anchor_mask + math.log(self.strength)
                appended_mask = torch.cat((inside_mask, anchor_mask), dim=-1)
            guided = _call_attention(original, q_in, appended_k, appended_v, heads, appended_mask)
        out[:, inside] = guided
        return out
