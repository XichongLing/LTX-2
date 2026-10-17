from __future__ import annotations

from .base import ChunkSpec, VideoChunker


def snap_down(value: int, lcm_constraint: int, base_offset: int) -> int:
    """Largest ``n <= value`` such that ``(n - base_offset) % lcm_constraint == 0``.

    Returns ``base_offset`` if ``value < base_offset`` (the smallest valid length).
    """
    if value <= base_offset:
        return base_offset
    return ((value - base_offset) // lcm_constraint) * lcm_constraint + base_offset


def snap_up(value: int, lcm_constraint: int, base_offset: int) -> int:
    """Smallest ``n >= value`` such that ``(n - base_offset) % lcm_constraint == 0``.

    Returns ``base_offset`` if ``value <= base_offset``.
    """
    if value <= base_offset:
        return base_offset
    remainder = (value - base_offset) % lcm_constraint
    if remainder == 0:
        return value
    return value + (lcm_constraint - remainder)


class UniformIntervalChunker(VideoChunker):
    """Naive chunker: emits segments of ``~target_len`` frames.

    Each segment length is snapped to ``lcm_constraint * k + base_offset`` so
    it satisfies LTX's latent frame-count constraint
    (``(num_frames - 1) % 8 == 0`` for the default ``lcm_constraint=8``,
    ``base_offset=1``).

    The trailing remainder is handled by snapping its span UP (not down) and,
    if that overshoots, shifting the chunk's ``start_frame`` backward so the
    chunk still ends exactly at ``total_frames``. This guarantees every chunk
    together covers ``[0, total_frames)`` -- no source frames are dropped.
    Any backward shift overlaps the previous chunk; the overlap length is
    recorded in ``overlap_in`` so the assembly stage can trim the duplicated
    frames before concatenation. If the remainder is shorter than
    ``min_chunk_frames``, it is merged into the previous chunk this way
    instead of becoming its own chunk.
    """

    def __init__(
        self,
        target_len: int,
        lcm_constraint: int = 8,
        base_offset: int = 1,
        min_chunk_frames: int | None = None,
    ) -> None:
        if target_len < base_offset:
            raise ValueError(f"target_len ({target_len}) must be >= base_offset ({base_offset})")
        if lcm_constraint <= 0:
            raise ValueError(f"lcm_constraint must be positive, got {lcm_constraint}")

        self.target_len = target_len
        self.lcm_constraint = lcm_constraint
        self.base_offset = base_offset
        self.chunk_len = snap_down(target_len, lcm_constraint, base_offset)
        self.min_chunk_frames = (
            min_chunk_frames if min_chunk_frames is not None else lcm_constraint + base_offset
        )

    def _snap_down(self, value: int) -> int:
        return snap_down(value, self.lcm_constraint, self.base_offset)

    def _snap_up(self, value: int) -> int:
        return snap_up(value, self.lcm_constraint, self.base_offset)

    def plan(self, total_frames: int, fps: float) -> list[ChunkSpec]:
        del fps  # unused by the uniform chunker; kept for interface symmetry

        if total_frames < self.base_offset:
            raise ValueError(
                f"total_frames ({total_frames}) is shorter than the minimum valid "
                f"chunk length ({self.base_offset})"
            )

        chunks: list[ChunkSpec] = []
        start = 0
        while total_frames - start >= self.chunk_len:
            chunks.append(ChunkSpec(index=len(chunks), start_frame=start, num_frames=self.chunk_len))
            start += self.chunk_len

        remaining = total_frames - start
        if remaining == 0:
            return chunks

        if chunks and self._snap_down(remaining) < self.min_chunk_frames:
            # Too small to stand on its own: fold it into the previous chunk.
            target_index = chunks[-1].index
            target_start = chunks[-1].start_frame
        else:
            target_index = len(chunks)
            target_start = start

        span = total_frames - target_start
        merged_len = self._snap_up(span)
        new_start = total_frames - merged_len
        if new_start < 0:
            # No earlier frames to borrow from (this is the first/only
            # chunk): fall back to snapping down, accepting the unavoidable
            # loss of up to lcm_constraint-1 trailing frames.
            merged_len = self._snap_down(span)
            new_start = target_start

        overlap = max(0, target_start - new_start)
        chunk = ChunkSpec(index=target_index, start_frame=new_start, num_frames=merged_len, overlap_in=overlap)
        if target_index < len(chunks):
            chunks[-1] = chunk
        else:
            chunks.append(chunk)

        return chunks
