#!/usr/bin/env python3
"""Argparse entry point (Section 9). Thin: builds a PipelineConfig and runs it.

LTX checkpoint/strength/attention-probe arguments mirror
``style_transfer_ltx23/translate_single_sequence.py`` so they can be forwarded
verbatim to ``build_style_transfer_command`` without re-declaring (and
drifting from) that surface (Section 4.4 / Section 9).
"""
from __future__ import annotations

import argparse

from .config import DEFAULT_PROMPT, DEFAULT_FLUX_MULTIREF_RUNNER, PipelineConfig
from .pipeline import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunked reference-guided style transfer: split a source video into "
            "temporal chunks, FLUX-edit each chunk's first frame toward a global "
            "appearance reference, run IC-LoRA style transfer per chunk, and "
            "concatenate the results."
        )
    )

    # Core I/O.
    parser.add_argument("--source-video", required=True, help="Source RGB video V0.")
    parser.add_argument("--target-reference", required=True, help="Global appearance/style target image I0.")
    parser.add_argument("--work-dir", required=True, help="Directory for chunk plan, intermediates, and manifest.json.")
    parser.add_argument("--output", required=True, help="Final concatenated stylized MP4 output path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Restyle prompt, shared by the FLUX edit and IC-LoRA stages.")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt text.")

    # Chunking (Section 6.1 / 4.2).
    parser.add_argument("--chunk-len", type=int, default=121, help="Target frames per chunk, snapped to an LTX-valid length.")
    parser.add_argument("--lcm-constraint", type=int, default=8, help="LTX latent temporal stride; valid lengths are lcm_constraint*k + base_offset.")
    parser.add_argument("--base-offset", type=int, default=1, help="LTX latent frame-count offset; valid lengths are lcm_constraint*k + base_offset.")
    parser.add_argument(
        "--divider",
        choices=("uniform",),
        default="uniform",
        help="Chunking strategy. Only 'uniform' (naive, non-overlapping) is implemented so far (Section 4.3).",
    )

    # FLUX two-reference edit passthrough (Section 6.2 / 7).
    parser.add_argument("--flux-runner-path", default=str(DEFAULT_FLUX_MULTIREF_RUNNER), help="Path to run_flux_multiref_edit.py.")
    parser.add_argument(
        "--flux-python",
        default=None,
        help=(
            "Python interpreter for the FLUX runner subprocess (must have `diffusers` installed; "
            "the LTX-2 venv does not). Defaults to experimental/.venv/bin/python if it exists, "
            "otherwise the current interpreter."
        ),
    )
    parser.add_argument("--flux-model", default=None, help="FLUX model id override.")
    parser.add_argument("--flux-seed", type=int, default=42)
    parser.add_argument("--flux-steps", type=int, default=50)
    parser.add_argument("--flux-guidance", type=float, default=4.0)
    parser.add_argument("--flux-device", default="cuda:0")
    parser.add_argument("--flux-no-cpu-offload", action="store_true", help="Disable CPU offload for the FLUX pipeline.")
    parser.add_argument("--flux-height", type=int, default=None, help="Override FLUX output height. Defaults to each chunk's first-frame height.")
    parser.add_argument("--flux-width", type=int, default=None, help="Override FLUX output width. Defaults to each chunk's first-frame width.")

    # LTX / IC-LoRA passthrough (forwarded to run_ic_lora_style_transfer.py via build_style_transfer_command).
    parser.add_argument("--distilled-checkpoint-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--ic-lora-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=None, help="Shared output width for every chunk (Section 10.4).")
    parser.add_argument("--height", type=int, default=None, help="Shared output height for every chunk (Section 10.4).")
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--image-frame-idx", type=int, default=0, help="Local frame index (within each chunk) where the edited first frame is injected. Should normally stay 0.")
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument(
        "--reference-image-replace",
        type=int,
        nargs="*",
        default=None,
        metavar="FRAME_IDX",
        help="Local frame indices for in-place latent replacement, applied per chunk. Defaults to frame 0.",
    )
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--quiet-layer-streaming", action="store_true")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None)
    parser.add_argument("--attention-probe-output", default=None)
    parser.add_argument("--attention-probe-layers", default="0,8,16,24,32,40,47")
    parser.add_argument("--attention-probe-steps", default="all")
    parser.add_argument("--attention-probe-heads", choices=("mean", "all"), default="mean")
    parser.add_argument("--attention-probe-query-chunk-size", type=int, default=128)
    parser.add_argument("--quantization", nargs="+", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = PipelineConfig.from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
