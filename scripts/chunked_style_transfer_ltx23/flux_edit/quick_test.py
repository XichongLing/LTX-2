#!/usr/bin/env python3
"""Ad-hoc single-image FLUX two-reference edit test ("eyeball edited first frames").

Wraps ``runner.edit_first_frame`` + ``prompt.build_multiref_prompt`` so the
Stage-B FLUX edit can be tried on two arbitrary images without going through
the full chunked pipeline.

Usage (from the LTX-2 venv):

    python -m chunked_style_transfer_ltx23.flux_edit.quick_test \\
        --target-reference I0.png --source-frame frame.png \\
        --prompt "..." --output out.png [--steps 25] [--no-cpu-offload]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .prompt import build_multiref_prompt
from .runner import edit_first_frame

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../LTX-2
DEFAULT_FLUX_MULTIREF_RUNNER = REPO_ROOT.parent / "flux.2_dev" / "run_flux_multiref_edit.py"
DEFAULT_FLUX_PYTHON = REPO_ROOT.parent / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-reference", required=True, help="Image 1: style/appearance reference (I0).")
    parser.add_argument("--source-frame", required=True, help="Image 2: scene to restyle.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--prompt", default="", help="User restyle instructions, appended to the role-disambiguation preamble.")
    parser.add_argument("--no-preamble", action="store_true", help="Use --prompt verbatim, skipping build_multiref_prompt's role-disambiguation preamble.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None, help="FLUX model id override.")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--flux-runner-path", default=str(DEFAULT_FLUX_MULTIREF_RUNNER))
    parser.add_argument("--flux-python", default=str(DEFAULT_FLUX_PYTHON))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_preamble:
        prompt = args.prompt.strip()
        if args.negative_prompt.strip():
            prompt = f"{prompt}\n\nNegative prompt: {args.negative_prompt.strip()}"
    else:
        prompt = build_multiref_prompt(args.prompt, args.negative_prompt)
    print(f"Prompt: {prompt}")

    out_path = edit_first_frame(
        target_reference=Path(args.target_reference),
        source_first_frame=Path(args.source_frame),
        out_path=Path(args.output),
        prompt=prompt,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        device=args.device,
        runner_path=Path(args.flux_runner_path),
        python=args.flux_python,
        model=args.model,
        cpu_offload=not args.no_cpu_offload,
        height=args.height,
        width=args.width,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
