#!/usr/bin/env python3
"""Video I/O helpers for the chunked style-transfer pipeline.

Re-exports the upstream frame/video I/O helpers from
``style_transfer_ltx23.style_transfer_preprocess`` and the frame-accurate
slicer from ``style_transfer_ltx23.translate_single_sequence_chunked`` (see
4.4 of the build plan: "Reuse by import, not copy"). ``slice_subvideos`` is
just a thin per-chunk loop over that slicer.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
STYLE_TRANSFER_DIR = SCRIPTS_DIR / "style_transfer_ltx23"
for _dir in (SCRIPTS_DIR, STYLE_TRANSFER_DIR):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)

from style_transfer_preprocess import (  # noqa: E402
    extract_first_frame_from_video,
    extract_frame_from_video,
    is_readable_video_file,
    read_video_metadata,
)
from translate_single_sequence_chunked import slice_video_by_frames  # noqa: E402

__all__ = [
    "extract_first_frame_from_video",
    "extract_frame_from_video",
    "is_readable_video_file",
    "read_video_metadata",
    "slice_subvideos",
    "trim_overlap",
]


def slice_subvideos(source_video: str | Path, chunks: list, work_dir: str | Path) -> list[Path]:
    """Slice ``source_video`` into one subvideo per chunk under ``work_dir/chunks/<dir_name>/source.mp4``.

    ``chunks`` is a list of ``chunking.base.ChunkSpec``. Each chunk is sliced
    independently via ``slice_video_by_frames`` (frame-accurate, re-encodes,
    reuses the output file if it already exists).
    """
    source_video = str(Path(source_video).expanduser().resolve())
    chunks_dir = Path(work_dir) / "chunks"

    outputs: list[Path] = []
    for chunk in chunks:
        out_path = chunks_dir / chunk.dir_name / "source.mp4"
        slice_video_by_frames(source_video, out_path, chunk.start_frame, chunk.end_frame)
        outputs.append(out_path)
    return outputs


def trim_overlap(video_path: str | Path, overlap_in: int, num_frames: int, out_path: str | Path, fps: float) -> Path:
    """Drop a chunk output's first ``overlap_in`` frames before concatenation.

    ``UniformIntervalChunker`` may shift a chunk's start backward so it still
    ends at ``total_frames`` (Section 4.3), overlapping the previous chunk by
    ``overlap_in`` frames. Trimming those duplicated frames here keeps the
    final concatenated video's length equal to the source video's.

    Per-chunk ``output.mp4`` files written by ``run_ic_lora_style_transfer.py``
    carry bogus frame-rate metadata (observed: ``num_frames * fps`` instead of
    ``fps``, e.g. ``1368/1`` for a 57-frame 24fps chunk). Passing the manifest's
    real ``fps`` as ``rate`` overrides that metadata so ``slice_video_by_frames``
    doesn't derive an absurd ``time_base`` that produces non-monotonic DTS on mux.
    """
    if overlap_in <= 0:
        return Path(video_path)
    return slice_video_by_frames(str(video_path), Path(out_path), overlap_in, num_frames, rate=int(round(fps)))
