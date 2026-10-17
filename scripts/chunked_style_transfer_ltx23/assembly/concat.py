#!/usr/bin/env python3
"""Stage D: concatenate per-chunk outputs into the final video (Section 6.4).

Reuses ``translate_single_sequence_chunked.concatenate_videos`` (Section 4.4):
re-encodes each chunk's frames onto a single output stream with sequential
pts, so the result is a single seamless-timebase MP4 regardless of any minor
encoder timestamp drift between chunks.

Naive non-overlapping concat (Section 4.3): no seam blending yet. That's left
to a future overlap/blend or carry-forward-conditioning divider.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # .../LTX-2/scripts
STYLE_TRANSFER_DIR = SCRIPTS_DIR / "style_transfer_ltx23"
for _dir in (SCRIPTS_DIR, STYLE_TRANSFER_DIR):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)

from style_transfer_preprocess import is_readable_video_file  # noqa: E402
from translate_single_sequence_chunked import concatenate_videos  # noqa: E402


def concat_chunks(chunk_outputs: list[Path], output_path: str | Path, fps: float) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    if is_readable_video_file(output_path):
        print(f"Reusing existing final video: {output_path}")
        return output_path

    if not chunk_outputs:
        raise ValueError("No chunk outputs to concatenate")

    concatenate_videos(chunk_outputs, output_path, fps)

    if not output_path.is_file():
        raise FileNotFoundError(f"concatenate_videos did not create {output_path}")
    return output_path
