from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSpec:
    """A single temporal segment of the source video.

    ``num_frames`` must already be an LTX-valid latent length, i.e.
    ``(num_frames - 1) % 8 == 0`` (see ``VIDEO_SCALE_FACTORS.time`` in
    ``ltx_core.model.video_vae``). Chunkers are responsible for producing
    only valid lengths; downstream stages assume this invariant holds.
    """

    index: int
    start_frame: int  # inclusive, in V0 frame space
    num_frames: int  # LTX-valid length: (num_frames - 1) % 8 == 0
    overlap_in: int = 0  # frames overlapping the previous chunk; trimmed before concat (assemble())

    @property
    def end_frame(self) -> int:
        """Exclusive end frame index in V0 frame space."""
        return self.start_frame + self.num_frames

    @property
    def dir_name(self) -> str:
        return f"chunk_{self.index:03d}"


class VideoChunker(ABC):
    """Splits a source video into a sequence of LTX-valid chunks."""

    @abstractmethod
    def plan(self, total_frames: int, fps: float) -> list[ChunkSpec]:
        """Return the ordered list of chunks covering ``total_frames``."""
        ...
