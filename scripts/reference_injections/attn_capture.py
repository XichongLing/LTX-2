"""
Capture cross-attention weights from LTX-2.3 during inference.

Primary strategy
----------------
Context manager that monkeypatches F.scaled_dot_product_attention.
Identifies cross-attention calls by k_len ∈ text_len_set (default {256}).
Recomputes softmax(Q·Kᵀ/√d) from the same Q, K — fused kernels don't expose
weights, but recomputing with the same inputs reproduces them exactly.

Fallback strategy
-----------------
install_module_hooks() registers forward-pre hooks on Attention modules
whose name contains `name_filter` (default "attn2") — use when flash-attn or
xformers is selected instead of SDPA (not the case on RTX 5090 / sm_12x).

Discovery mode
--------------
Set discover=True on AttentionStore to log every (q_len, k_len) pair so you
can confirm the text length and which call indices are cross-attn.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional, Set

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AttentionRecord:
    step: int        # denoising step index (0-based, incremented by mark_step)
    call_idx: int    # SDPA call index within this step (reset by mark_step)
    q_len: int
    k_len: int
    # shape: (heads, q_len, k_len) on CPU fp16
    weights: torch.Tensor


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AttentionStore:
    """Accumulates cross-attention weights over denoising steps."""

    def __init__(
        self,
        text_len_set: Set[int] = frozenset({256}),
        batch_index: Optional[int] = None,   # None → mean over batch dim
        discover: bool = False,              # log all (q_len, k_len) pairs
    ) -> None:
        self._text_len_set = set(text_len_set)
        self._batch_index = batch_index
        self._discover = discover
        self._step: int = -1   # incremented to 0 on first mark_step()
        self._call_idx: int = 0
        self.records: list[AttentionRecord] = []
        self._discover_log: list[dict] = []

    # ------------------------------------------------------------------
    def mark_step(self) -> None:
        """Call once at the start of each denoising step."""
        self._step += 1
        self._call_idx = 0

    def reset(self) -> None:
        self._step = -1
        self._call_idx = 0
        self.records.clear()
        self._discover_log.clear()

    # ------------------------------------------------------------------
    def _ingest(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_mask=None,
        scale=None,
        is_causal: bool = False,
    ) -> None:
        """
        Called inside the SDPA wrapper.  NEVER raises.

        Obtains attention weights by calling SDPA a second time with V = I_(k_len),
        so the output equals softmax(QKᵀ/√d) directly.  This reuses the fused
        kernel (FlashAttention/efficient-attn) and avoids materialising a separate
        (B*H, q, k) intermediate tensor on GPU.

        Output (B, H, q_len, k_len) is averaged over B×H before storing, so each
        record is (q_len, k_len) fp16 ≈ 12.8 MB.  768 records × 12.8 MB = 9.8 GB
        CPU RAM — feasible.
        """
        try:
            k_len = k.shape[-2]

            if self._discover:
                self._discover_log.append({
                    "step": self._step,
                    "call_idx": self._call_idx,
                    "q_len": q.shape[-2],
                    "k_len": k_len,
                })
                return   # discover = metadata only, never compute weights

            if k_len not in self._text_len_set:
                return

            # Reject text self-attention (Gemma encoder): q_len == k_len.
            # We only want video→text cross-attention where q_len >> k_len.
            if q.shape[-2] <= k_len:
                return

            # Normalise to 4-D (B, H, seq, d)
            if q.ndim == 3:
                q4 = q.unsqueeze(1)
                k4 = k.unsqueeze(1)
            else:
                q4 = q
                k4 = k

            B, H, q_len, d = q4.shape

            # Identity V: (1, 1, k_len, k_len) — broadcastable over B and H.
            # SDPA output = softmax(QKᵀ/√d) @ I = softmax(QKᵀ/√d) = weight matrix.
            eye_v = torch.eye(k_len, dtype=q4.dtype, device=q4.device).unsqueeze(0).unsqueeze(0)

            extra = {"scale": scale} if scale is not None else {}
            with torch.no_grad():
                weights_bhqk = _ORIG_SDPA(
                    q4, k4, eye_v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    **extra,
                )   # (B, H, q_len, k_len)

            # Average over B×H before moving to CPU to keep per-record size small.
            if self._batch_index is not None:
                w = weights_bhqk[self._batch_index].mean(0)   # (q_len, k_len)
            else:
                w = weights_bhqk.mean(dim=(0, 1))              # (q_len, k_len)

            self.records.append(AttentionRecord(
                step=self._step,
                call_idx=self._call_idx,
                q_len=q_len,
                k_len=k_len,
                weights=w.cpu().half(),
            ))
        except Exception:
            pass   # capture errors must never break the forward

    # ------------------------------------------------------------------
    def print_discover_table(self) -> None:
        if not self._discover_log:
            print("No calls captured. Ensure capture() was active.")
            return

        # Unique (q_len, k_len) pairs with count and first call_idx
        from collections import defaultdict
        summary: dict = {}
        for e in self._discover_log:
            key = (e["q_len"], e["k_len"])
            if key not in summary:
                summary[key] = {"count": 0, "first_call": e["call_idx"]}
            summary[key]["count"] += 1

        print(f"\n{'q_len':>8}  {'k_len':>8}  {'count':>6}  {'first_call':>10}  note")
        print("─" * 56)
        for (q_len, k_len), info in sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True):
            note = ""
            if k_len in {128, 256, 512, 1024}:
                note = "← TEXT cross-attn candidate"
            elif q_len == k_len:
                note = "(self-attn)"
            print(f"{q_len:>8}  {k_len:>8}  {info['count']:>6}  {info['first_call']:>10}  {note}")
        print(f"\n{len(self._discover_log)} calls total  |  {len(summary)} unique (q_len, k_len) pairs")


# ---------------------------------------------------------------------------
# Primary: SDPA monkeypatch
# ---------------------------------------------------------------------------

_ORIG_SDPA = None


@contextlib.contextmanager
def capture(store: AttentionStore):
    """
    Patch F.scaled_dot_product_attention for the duration of the with-block.
    Restores the original on exit even if an exception occurs.
    """
    global _ORIG_SDPA
    _ORIG_SDPA = F.scaled_dot_product_attention

    def _patched(
        query, key, value,
        attn_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
        **kwargs,
    ):
        # Always call original first — output is unchanged
        extra = {"scale": scale} if scale is not None else {}
        out = _ORIG_SDPA(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **extra,
            **kwargs,
        )
        store._ingest(query, key, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        store._call_idx += 1
        return out

    F.scaled_dot_product_attention = _patched
    try:
        yield store
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA


# ---------------------------------------------------------------------------
# Fallback: module forward-pre hooks
# ---------------------------------------------------------------------------

def install_module_hooks(
    model: torch.nn.Module,
    store: AttentionStore,
    name_filter: str = "attn2",
) -> list:
    """
    Register forward-pre hooks on Attention modules whose full qualified name
    contains `name_filter`.  Returns hook handles; call remove_hooks() to clean up.

    Hooked modules are the Attention instances in transformer.py:
        self.attn2       — video cross-attention (context_dim=4096)
        self.audio_attn2 — audio cross-attention (context_dim=2048)

    Adjust if the transformer block structure changes.
    """
    handles = []

    def _pre_hook(module, args, kwargs):
        """
        Attention.forward(x, context=..., ...) — intercept to get Q, K.
        x: (B, T, query_dim)
        context: (B, S, context_dim)  — S=256 for text cross-attn
        """
        try:
            x = args[0] if args else kwargs.get("x") or kwargs.get("hidden_states")
            context = kwargs.get("context") or (args[1] if len(args) > 1 else None)
            if context is None or x is None:
                return
            with torch.no_grad():
                q = module.to_q(x)
                k = module.to_k(context)
            store._ingest(q, k)
            store._call_idx += 1
        except Exception:
            pass

    for name, module in model.named_modules():
        if name_filter in name and hasattr(module, "to_q") and hasattr(module, "to_k"):
            h = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            handles.append(h)

    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
