#!/usr/bin/env python3
"""PipelineConfig: typed view over the parsed CLI args used by pipeline.py."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .chunking.base import VideoChunker
from .chunking.uniform import UniformIntervalChunker

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../LTX-2
DEFAULT_FLUX_MULTIREF_RUNNER = REPO_ROOT.parent / "flux.2_dev" / "run_flux_multiref_edit.py"
# flux.2_dev has its own venv (with `diffusers`) at the experimental/ root,
# separate from LTX-2/.venv. Used as the --flux-python default when present
# (see flux.2_dev/run_low_fidelity_fixed_photoreal.sh for the same convention).
DEFAULT_FLUX_PYTHON = REPO_ROOT.parent / ".venv" / "bin" / "python"

DEFAULT_PROMPT = (
    "Restyle the scene to match the appearance, materials, colors, and lighting of the "
    "target reference image, while preserving the motion, camera movement, and layout of "
    "the source video."
)


@dataclass
class PipelineConfig:
    source_video: Path
    target_reference: Path
    work_dir: Path
    output: Path
    chunker: VideoChunker

    user_prompt: str
    negative_prompt: str

    flux_runner_path: Path
    flux_python: str | None
    flux_model: str | None
    flux_seed: int
    flux_steps: int
    flux_guidance: float
    flux_device: str
    flux_cpu_offload: bool
    flux_height: int | None
    flux_width: int | None

    # Forwarded verbatim to translate_single_sequence.build_style_transfer_command.
    ltx_args: argparse.Namespace

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        chunker = UniformIntervalChunker(
            target_len=args.chunk_len,
            lcm_constraint=args.lcm_constraint,
            base_offset=args.base_offset,
        )

        ltx_args = argparse.Namespace(**vars(args))
        ltx_args.prompt = args.prompt
        ltx_args.negative_prompt = args.negative_prompt
        # Per-chunk runner always feeds the chunk subvideo as both reference and
        # conditioning video in rgb mode (Section 6.3); pin it here so the
        # passthrough namespace can't drift from that contract.
        ltx_args.conditioning_mode = "rgb"

        flux_python = args.flux_python
        if flux_python is None and DEFAULT_FLUX_PYTHON.is_file():
            flux_python = str(DEFAULT_FLUX_PYTHON)

        return cls(
            source_video=Path(args.source_video).expanduser().resolve(),
            target_reference=Path(args.target_reference).expanduser().resolve(),
            work_dir=Path(args.work_dir).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            chunker=chunker,
            user_prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            flux_runner_path=Path(args.flux_runner_path).expanduser().resolve(),
            flux_python=flux_python,
            flux_model=args.flux_model,
            flux_seed=args.flux_seed,
            flux_steps=args.flux_steps,
            flux_guidance=args.flux_guidance,
            flux_device=args.flux_device,
            flux_cpu_offload=not args.flux_no_cpu_offload,
            flux_height=args.flux_height,
            flux_width=args.flux_width,
            ltx_args=ltx_args,
        )
