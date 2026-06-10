"""
Pure-tensor/numpy post-processing: attention records → spatial mask.

No LTX import — fully unit-testable offline.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Token discovery
# ---------------------------------------------------------------------------

def find_token_indices(tokenizer, prompt: str, word: str) -> List[int]:
    """
    Best-effort: 0-based indices in the padded text token sequence for `word`.

    Tries fast-tokenizer offset mapping, then falls back to token-id subsequence
    search.  This is fragile — use sweep mode to verify by eye.
    """
    word_lower = word.lower()

    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        indices = [
            i for i, (s, e) in enumerate(enc["offset_mapping"])
            if s < e and word_lower in prompt[s:e].lower()
        ]
        if indices:
            return indices
    except Exception:
        pass

    try:
        p_ids: List[int] = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        w_ids: List[int] = tokenizer(word, add_special_tokens=False)["input_ids"]
        n = len(w_ids)
        return [
            i + j
            for i in range(len(p_ids) - n + 1)
            if p_ids[i : i + n] == w_ids
            for j in range(n)
        ]
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    records: list,
    call_indices: Optional[Sequence[int]] = None,
    steps: Optional[Sequence[int]] = None,
    target_q_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Mean over selected records then over heads.  Returns (q_len, k_len) float32.

    call_indices: which call_idx values to include (None = all)
    steps:        which step indices to include (None = all)
    target_q_len: filter to this q_len; None = use the modal q_len
    """
    if not records:
        raise ValueError("No records — was capture() active?")

    filt = list(records)
    if call_indices is not None:
        cs = set(call_indices)
        filt = [r for r in filt if r.call_idx in cs]
    if steps is not None:
        ss = set(steps)
        filt = [r for r in filt if r.step in ss]
    if not filt:
        raise ValueError("No records remain after filtering.")

    if target_q_len is None:
        target_q_len = Counter(r.q_len for r in filt).most_common(1)[0][0]

    filt = [r for r in filt if r.q_len == target_q_len]
    if not filt:
        raise ValueError(f"No records with q_len={target_q_len}.")

    # Running mean avoids materialising (N, q, k) all at once.
    # With N=768, q=6240, k=1024 that would be ~19.7 GB float32.
    acc = filt[0].weights.float().clone()
    for r in filt[1:]:
        acc += r.weights.float()
    return acc / len(filt)


# ---------------------------------------------------------------------------
# Token map
# ---------------------------------------------------------------------------

def token_map(agg: torch.Tensor, token_indices: Sequence[int]) -> torch.Tensor:
    """
    Sum the attention columns for the target word tokens.
    agg: (q_len, k_len) → returns (q_len,).
    """
    return agg[:, list(token_indices)].sum(dim=-1)


# ---------------------------------------------------------------------------
# Reshape to spatial grid
# ---------------------------------------------------------------------------

def reshape_to_grid(
    vec: torch.Tensor,
    grid_thw: Tuple[int, int, int],
    n_target_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Reshape attention-score vector to (T, H, W) in LATENT space.

    ⚠ grid_thw must be the ACTUAL latent dimensions from the pipeline output
    (e.g. latent.shape[2:]). Do NOT compute from pixel dims as H_px // 32
    because the VAE encoder uses ceil-mode convolutions when H_px is not
    divisible by 32, so the actual H_lat may differ by ±1 from H_px // 32.
    Always get grid_thw from the returned video_state.latent.shape.

    Token order (frame-major, row-major, col-major):
        flat_i = t * H * W + h * W + w
    This matches VideoLatentPatchifier.patchify() via einops:
        "b c (f p1) (h p2) (w p3) -> b (f h w) ..."
    Verify against components/patchifiers.py if the overlay looks scrambled.

    IC-LoRA note: the transformer sequence is [target | reference] tokens.
    Pass n_target_tokens = T * H * W to use only the target half.
    """
    T, H, W = grid_thw
    n = T * H * W

    if n_target_tokens is not None:
        vec = vec[:n_target_tokens]

    if vec.shape[0] < n:
        raise ValueError(
            f"vec length {vec.shape[0]} < T*H*W={n}. "
            f"For IC-LoRA pass n_target_tokens={n}."
        )
    return vec[:n].reshape(T, H, W)


# ---------------------------------------------------------------------------
# Upsample + threshold
# ---------------------------------------------------------------------------

def to_pixel_mask(
    grid: torch.Tensor,
    out_hw: Tuple[int, int],
    threshold: str = "quantile",
    keep_frac: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear-upsample (T, H_lat, W_lat) to pixel resolution, per-frame
    min-max normalise, threshold.

    out_hw: actual pixel resolution (H_px, W_px) — independent of divisibility
            since this is a pure interpolation, not formula-based.

    Returns:
        heatmap: (T, H_px, W_px) float32 in [0,1], per-frame normalised.
        mask:    (T, H_px, W_px) bool.
    """
    import torch.nn.functional as Fn

    T = grid.shape[0]
    out_H, out_W = out_hw

    frames_up = []
    for t in range(T):
        frame = grid[t].unsqueeze(0).unsqueeze(0).float()
        up = Fn.interpolate(frame, size=(out_H, out_W), mode="bilinear", align_corners=False)
        frames_up.append(up.squeeze(0).squeeze(0))
    heatmap_raw = torch.stack(frames_up, dim=0)   # (T, H_px, W_px)

    flat = heatmap_raw.reshape(T, -1)
    mn = flat.min(dim=1).values.reshape(T, 1, 1)
    mx = flat.max(dim=1).values.reshape(T, 1, 1)
    heatmap = (heatmap_raw - mn) / (mx - mn + 1e-6)

    masks = []
    for t in range(T):
        h_t = heatmap[t]
        if threshold == "quantile":
            tv = torch.quantile(h_t.flatten().float(), 1.0 - keep_frac)
            masks.append(h_t >= tv)
        elif threshold == "otsu":
            masks.append(_otsu(h_t))
        elif threshold == "abs":
            masks.append(h_t >= keep_frac)
        else:
            raise ValueError(f"Unknown threshold '{threshold}'. Use quantile/otsu/abs.")

    return heatmap, torch.stack(masks, dim=0)


def _otsu(img: torch.Tensor) -> torch.Tensor:
    arr = img.cpu().float().numpy().flatten()
    counts, edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
    total = arr.size
    w_bg, sum_bg = 0, 0.0
    sum_total = float(np.dot(np.arange(256), counts))
    best_var, best_t = 0.0, 0.0
    for i in range(256):
        w_bg += counts[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += i * counts[i]
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > best_var:
            best_var, best_t = var, edges[i + 1]
    return img >= best_t


# ---------------------------------------------------------------------------
# Offline unit test — no model required
# ---------------------------------------------------------------------------

def _unit_test() -> None:
    """
    Plant a known attention blob on text-token column 10, run the full
    pipeline (aggregate → token_map → reshape → to_pixel_mask), and assert
    the recovered mask covers the planted region.

    Run:  python mask_from_attn.py
    """
    from dataclasses import dataclass

    @dataclass
    class FakeRecord:
        step: int
        call_idx: int
        q_len: int
        k_len: int
        weights: torch.Tensor

    # Arbitrary latent grid — intentionally NOT multiples of 32 to verify
    # that the pipeline is grid-space math only (upscaling handles the rest)
    T, H, W = 1, 7, 9
    q_len = T * H * W  # 63
    k_len = 256

    base = torch.ones(q_len, k_len) * 0.005
    # Plant a blob: latent tokens at h=3..5, w=4..6 on text token 10
    blob = [3 * W + 4, 3 * W + 5, 3 * W + 6,
            4 * W + 4, 4 * W + 5, 4 * W + 6,
            5 * W + 4, 5 * W + 5, 5 * W + 6]
    base[blob, 10] = 1.0

    records = [
        FakeRecord(step=s, call_idx=0, q_len=q_len, k_len=k_len,
                   weights=base.clone())
        for s in range(8)
    ]

    agg = aggregate(records)
    assert agg.shape == (q_len, k_len)

    vec = token_map(agg, [10])
    assert vec.shape == (q_len,)

    # Use the grid dims directly (not H*32 — intentional)
    grid = reshape_to_grid(vec, (T, H, W))
    assert grid.shape == (T, H, W)

    # Upsample to some pixel resolution (not divisible by grid dims — intentional)
    px_H, px_W = 400, 500
    heat, mask = to_pixel_mask(grid, (px_H, px_W), threshold="quantile", keep_frac=0.2)
    assert heat.shape == (T, px_H, px_W)
    assert mask.shape == (T, px_H, px_W)

    # Check coverage in the blob region (latent h=3..5, w=4..6 → scaled region)
    rh = slice(int(3 / H * px_H), int(6 / H * px_H))
    rw = slice(int(4 / W * px_W), int(7 / W * px_W))
    coverage = mask[0, rh, rw].float().mean().item()
    assert coverage > 0.7, f"Blob coverage {coverage:.2f} < 0.7"

    print(
        f"[unit test] PASSED  grid={T}×{H}×{W}  pixel={px_H}×{px_W}  "
        f"blob_coverage={coverage:.2f}  heat=[{heat.min():.3f},{heat.max():.3f}]  "
        f"mask_on={mask.sum().item()}/{mask.numel()}"
    )

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        import overlay as ov
        frame = np.zeros((px_H, px_W, 3), dtype=np.uint8)
        result = ov.overlay_frame(frame, heat[0].numpy(), mask[0].numpy())
        assert result.shape == (px_H, px_W, 3)
        print("[unit test] overlay integration: PASSED")
    except ImportError:
        print("[unit test] overlay.py not found — skipping overlay check.")


if __name__ == "__main__":
    _unit_test()
