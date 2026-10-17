#!/usr/bin/env python3
"""Stage B: per-chunk FLUX.2 two-reference edit (Section 6.2 / 4.1).

Invokes the ``run_flux_multiref_edit.py`` sibling runner (see
``flux.2_dev/run_flux_multiref_edit.py``) as a subprocess so this package only
depends on that runner's CLI, not its internals (Section 4.4).

``edit_first_frames_batch`` is the Stage-B entry point used by the
orchestrator: it writes one job per chunk to a JSON file and invokes the
runner once, so the FLUX pipeline is loaded a single time for all N chunks
(Section 4.1). ``edit_first_frame`` is a single-job convenience wrapper
around the same batch path, useful for ad-hoc runs (Phase 1: "eyeball edited
first frames").
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


@dataclass(frozen=True)
class FluxEditJob:
    target_reference: Path  # I0 -- global appearance target ("image 1")
    source_first_frame: Path  # firstframe(s_i) -- structure to restyle ("image 2")
    output: Path
    prompt: str

    def to_json_entry(self) -> dict:
        return {
            "references": [str(self.target_reference), str(self.source_first_frame)],
            "output": str(self.output),
            "prompt": self.prompt,
        }


def _common_args(
    *,
    seed: int,
    steps: int,
    guidance: float,
    device: str,
    model: str | None,
    cpu_offload: bool,
    height: int | None,
    width: int | None,
) -> list[str]:
    args = [
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--guidance",
        str(guidance),
        "--device",
        device,
    ]
    if model:
        args.extend(["--model", model])
    if not cpu_offload:
        args.append("--no-cpu-offload")
    if height is not None:
        args.extend(["--height", str(height)])
    if width is not None:
        args.extend(["--width", str(width)])
    return args


def edit_first_frames_batch(
    jobs: list[FluxEditJob],
    *,
    seed: int = 42,
    steps: int = 50,
    guidance: float = 4.0,
    device: str = "cuda:0",
    runner_path: Path,
    python: str | None = None,
    model: str | None = None,
    cpu_offload: bool = True,
    height: int | None = None,
    width: int | None = None,
    work_dir: Path,
) -> list[Path]:
    """Run all ``jobs`` through one FLUX pipeline load (Section 4.1, Stage B).

    Jobs whose ``output`` already exists are left to the runner's
    reuse-if-exists check. If every job's output already exists, the
    subprocess is skipped entirely.
    """
    outputs = [job.output for job in jobs]
    for job in jobs:
        Path(job.output).parent.mkdir(parents=True, exist_ok=True)

    if all(Path(output).is_file() for output in outputs):
        print(f"Reusing existing edited first frames for all {len(jobs)} chunk(s).")
        return outputs

    runner_path = Path(runner_path).expanduser().resolve()
    if not runner_path.is_file():
        raise FileNotFoundError(
            f"FLUX multi-reference runner not found: {runner_path}. "
            "Pass --flux-runner-path to point at run_flux_multiref_edit.py."
        )

    jobs_path = Path(work_dir) / "flux_jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps([job.to_json_entry() for job in jobs], indent=2) + "\n", encoding="utf-8")

    command = [
        python or sys.executable,
        str(runner_path),
        "--jobs-json",
        str(jobs_path),
        *_common_args(
            seed=seed,
            steps=steps,
            guidance=guidance,
            device=device,
            model=model,
            cpu_offload=cpu_offload,
            height=height,
            width=width,
        ),
    ]
    run_command(command)

    missing = [output for output in outputs if not Path(output).is_file()]
    if missing:
        raise FileNotFoundError(f"The FLUX multi-reference runner did not create: {missing}")
    return outputs


def edit_first_frame(
    target_reference: Path,
    source_first_frame: Path,
    out_path: Path,
    prompt: str,
    *,
    seed: int = 42,
    steps: int = 50,
    guidance: float = 4.0,
    device: str = "cuda:0",
    runner_path: Path,
    python: str | None = None,
    model: str | None = None,
    cpu_offload: bool = True,
    height: int | None = None,
    width: int | None = None,
) -> Path:
    """Edit a single first frame. Convenience wrapper around ``edit_first_frames_batch``.

    ``target_reference`` is "image 1" (the global appearance target, I0) and
    ``source_first_frame`` is "image 2" (the per-chunk structure reference);
    ``prompt`` should already disambiguate these roles (see
    ``flux_edit.prompt.build_multiref_prompt``).
    """
    out_path = Path(out_path)
    job = FluxEditJob(
        target_reference=Path(target_reference).expanduser().resolve(),
        source_first_frame=Path(source_first_frame).expanduser().resolve(),
        output=out_path,
        prompt=prompt,
    )
    with tempfile.TemporaryDirectory(prefix="flux_edit_single_") as tmp_dir:
        edit_first_frames_batch(
            [job],
            seed=seed,
            steps=steps,
            guidance=guidance,
            device=device,
            runner_path=runner_path,
            python=python,
            model=model,
            cpu_offload=cpu_offload,
            height=height,
            width=width,
            work_dir=Path(tmp_dir),
        )
    return out_path


__all__ = ["FluxEditJob", "edit_first_frame", "edit_first_frames_batch"]
