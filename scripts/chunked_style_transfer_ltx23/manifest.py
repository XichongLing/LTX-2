#!/usr/bin/env python3
"""Manifest read/write for the chunked style-transfer work directory.

The manifest records the chunk plan and the per-chunk artifact paths so each
stage (Section 4.1: chunk / FLUX / LTX / concat) can run independently and be
resumed. Actual "is this artifact done" checks remain file-existence based
(matching the upstream "reuse if file exists" convention); the manifest is
the durable record of the plan plus a convenient index of artifact paths.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking.base import ChunkSpec

MANIFEST_FILENAME = "manifest.json"


@dataclass
class ChunkRecord:
    index: int
    start_frame: int
    num_frames: int
    overlap_in: int = 0
    source_video: str | None = None
    first_frame: str | None = None
    edited_first_frame: str | None = None
    output_video: str | None = None

    @property
    def dir_name(self) -> str:
        return f"chunk_{self.index:03d}"

    def to_spec(self) -> ChunkSpec:
        return ChunkSpec(
            index=self.index,
            start_frame=self.start_frame,
            num_frames=self.num_frames,
            overlap_in=self.overlap_in,
        )

    @classmethod
    def from_spec(cls, spec: ChunkSpec) -> "ChunkRecord":
        return cls(
            index=spec.index,
            start_frame=spec.start_frame,
            num_frames=spec.num_frames,
            overlap_in=spec.overlap_in,
        )


@dataclass
class Manifest:
    source_video: str
    target_reference: str
    fps: float
    total_frames: int
    chunks: list[ChunkRecord] = field(default_factory=list)

    @classmethod
    def from_chunks(
        cls,
        source_video: str | Path,
        target_reference: str | Path,
        fps: float,
        total_frames: int,
        chunks: list[ChunkSpec],
    ) -> "Manifest":
        return cls(
            source_video=str(source_video),
            target_reference=str(target_reference),
            fps=fps,
            total_frames=total_frames,
            chunks=[ChunkRecord.from_spec(spec) for spec in chunks],
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        payload = json.loads(text)
        chunk_payloads = payload.pop("chunks", [])
        chunks = [ChunkRecord(**chunk_payload) for chunk_payload in chunk_payloads]
        return cls(chunks=chunks, **payload)


def manifest_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / MANIFEST_FILENAME


def write_manifest(work_dir: str | Path, manifest: Manifest) -> Path:
    path = manifest_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def read_manifest(work_dir: str | Path) -> Manifest | None:
    path = manifest_path(work_dir)
    if not path.is_file():
        return None
    return Manifest.from_json(path.read_text(encoding="utf-8"))


def chunk_dir(work_dir: str | Path, record: ChunkRecord) -> Path:
    return Path(work_dir) / "chunks" / record.dir_name
