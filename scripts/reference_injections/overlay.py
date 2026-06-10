"""
Visualisation helpers: attention heatmaps and binary masks onto video frames.

No LTX import — fully unit-testable offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Per-frame composite
# ---------------------------------------------------------------------------

def overlay_frame(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    mask: Optional[np.ndarray] = None,
    heat_alpha: float = 0.55,
    mask_color_bgr: Tuple[int, int, int] = (0, 60, 255),
    mask_border_only: bool = False,
) -> np.ndarray:
    """
    Composite heatmap (and optionally mask contour) over a BGR frame.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR
        heatmap:   (H, W) float32 in [0, 1], same spatial size as frame
        mask:      (H, W) bool — optional; draws contour or fill
        heat_alpha: how strongly the jet-colormap heatmap is blended in
        mask_color_bgr: BGR colour for mask overlay
        mask_border_only: if True draw contour only; else filled semi-transparent

    Returns:
        (H, W, 3) uint8 BGR composite
    """
    out = frame_bgr.astype(np.float32)

    # Jet-colormap heatmap
    h_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    jet = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET).astype(np.float32)
    out = cv2.addWeighted(out, 1.0 - heat_alpha, jet, heat_alpha, 0)

    if mask is not None:
        mask_u8 = mask.astype(np.uint8) * 255
        if mask_border_only:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, mask_color_bgr, 2)
        else:
            overlay = np.full_like(out, mask_color_bgr, dtype=np.float32)
            alpha_mask = (mask.astype(np.float32) * 0.35)[..., None]
            out = out * (1.0 - alpha_mask) + overlay * alpha_mask

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-frame writers
# ---------------------------------------------------------------------------

def write_heatmap_frames(
    frames_bgr: Sequence[np.ndarray],
    heatmaps: np.ndarray,
    masks: Optional[np.ndarray],
    out_dir: Path,
    prefix: str = "frame",
    **overlay_kw,
) -> None:
    """
    Write one PNG per frame to out_dir.

    frames_bgr: list of (H, W, 3) uint8 frames
    heatmaps:   (T, H, W) float32
    masks:      (T, H, W) bool | None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(frames_bgr)
    for t in range(T):
        comp = overlay_frame(
            frames_bgr[t],
            heatmaps[t],
            masks[t] if masks is not None else None,
            **overlay_kw,
        )
        cv2.imwrite(str(out_dir / f"{prefix}_{t:04d}.png"), comp)


def write_heatmap_video(
    frames_bgr: Sequence[np.ndarray],
    heatmaps: np.ndarray,
    masks: Optional[np.ndarray],
    out_path: str | Path,
    fps: float = 24.0,
    **overlay_kw,
) -> None:
    """Write composited frames as an mp4 video."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(frames_bgr)
    if T == 0:
        return
    H, W = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    try:
        for t in range(T):
            comp = overlay_frame(
                frames_bgr[t],
                heatmaps[t],
                masks[t] if masks is not None else None,
                **overlay_kw,
            )
            writer.write(comp)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Sweep output: token-indexed grid image
# ---------------------------------------------------------------------------

def write_sweep_grid(
    frame_bgr: np.ndarray,
    token_heatmaps: np.ndarray,
    token_labels: Sequence[str],
    out_path: str | Path,
    cols: int = 8,
    thumb_h: int = 120,
) -> None:
    """
    One big image: a grid of frame+heatmap thumbnails, one per active text token.

    frame_bgr:      (H, W, 3) reference frame
    token_heatmaps: (K, H, W) float32, one heatmap per token
    token_labels:   K labels (e.g. "42: fountain")
    out_path:       save destination (.png)
    """
    K = len(token_heatmaps)
    if K == 0:
        return
    H, W = frame_bgr.shape[:2]
    aspect = W / H
    thumb_w = int(thumb_h * aspect)
    rows = (K + cols - 1) // cols
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i in range(K):
        row, col = divmod(i, cols)
        comp = overlay_frame(frame_bgr, token_heatmaps[i])
        thumb = cv2.resize(comp, (thumb_w, thumb_h))
        label = token_labels[i] if i < len(token_labels) else str(i)
        cv2.putText(thumb, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        canvas[row * thumb_h:(row + 1) * thumb_h, col * thumb_w:(col + 1) * thumb_w] = thumb
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# Video I/O helper
# ---------------------------------------------------------------------------

def read_video_frames_bgr(video_path: str | Path) -> Tuple[list, float]:
    """
    Returns (frames_bgr, fps) where frames_bgr is a list of (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps
