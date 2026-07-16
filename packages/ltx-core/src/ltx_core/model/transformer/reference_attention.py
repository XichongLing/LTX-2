from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch


Mode = Literal["capture", "inject"]


@dataclass
class ReferenceKV:
    """Reference attention bank for one transformer layer."""

    k: torch.Tensor
    v: torch.Tensor


@dataclass
class ReferenceAttentionController:
    """Runtime state for training-free reference K/V capture and injection.

    The controller is intentionally free of pipeline assumptions. A caller runs a
    reference forward inside ``mode="capture"``, then a source forward inside
    ``mode="inject"`` with the same controller.
    """

    mode: Mode
    layers: set[int]
    strength: float = 0.15
    attention_kind: str = "video_self"
    ref_token_mask: torch.Tensor | None = None
    ref_token_downscale: int = 1
    bank: dict[int, ReferenceKV] = field(default_factory=dict)
    captured_shapes: dict[int, tuple[int, ...]] = field(default_factory=dict)
    injected_shapes: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def wants(self, layer_idx: int | None, attention_kind: str | None) -> bool:
        return (
            layer_idx is not None
            and attention_kind == self.attention_kind
            and layer_idx in self.layers
        )

    def capture(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        k_ref = self._select_reference_tokens(k.detach())
        v_ref = self._select_reference_tokens(v.detach())
        self.bank[layer_idx] = ReferenceKV(k=k_ref, v=v_ref)
        self.captured_shapes[layer_idx] = tuple(k_ref.shape)

    def maybe_inject(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ref = self.bank.get(layer_idx)
        if ref is None or self.strength <= 0:
            return k, v, mask

        k_ref = self._match_batch(ref.k, k.shape[0]).to(device=k.device, dtype=k.dtype)
        v_ref = self._match_batch(ref.v, v.shape[0]).to(device=v.device, dtype=v.dtype)
        if self.strength != 1.0:
            k_ref = k_ref * self.strength

        k_out = torch.cat([k, k_ref], dim=1)
        v_out = torch.cat([v, v_ref], dim=1)
        mask_out = _expand_attention_mask(mask, extra_k_len=k_ref.shape[1])
        self.injected_shapes[layer_idx] = tuple(k_out.shape)
        return k_out, v_out, mask_out

    def _select_reference_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        if self.ref_token_mask is not None:
            token_mask = self.ref_token_mask.to(device=out.device, dtype=torch.bool).flatten()
            if token_mask.numel() != out.shape[1]:
                raise ValueError(
                    "ref_token_mask length must match the reference attention sequence length: "
                    f"{token_mask.numel()} != {out.shape[1]}"
                )
            out = out[:, token_mask]
        if self.ref_token_downscale > 1:
            out = out[:, :: self.ref_token_downscale]
        return out.contiguous()

    @staticmethod
    def _match_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch, -1, -1)
        raise ValueError(f"Reference K/V batch {tensor.shape[0]} cannot be broadcast to source batch {batch}")


_ACTIVE_CONTROLLER: ContextVar[ReferenceAttentionController | None] = ContextVar(
    "reference_attention_controller",
    default=None,
)


def get_reference_attention_controller() -> ReferenceAttentionController | None:
    return _ACTIVE_CONTROLLER.get()


@contextmanager
def reference_attention(controller: ReferenceAttentionController | None) -> Iterator[None]:
    token = _ACTIVE_CONTROLLER.set(controller)
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER.reset(token)


def _expand_attention_mask(mask: torch.Tensor | None, extra_k_len: int) -> torch.Tensor | None:
    if mask is None or extra_k_len == 0:
        return mask
    pad_shape = list(mask.shape)
    pad_shape[-1] = extra_k_len
    zeros = torch.zeros(pad_shape, dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, zeros], dim=-1)
