#!/usr/bin/env python3
"""Stage C: per-chunk IC-LoRA style transfer (Section 6.3).

Wraps ``run_ic_lora_style_transfer.py`` by reusing
``translate_single_sequence.build_style_transfer_command`` (Section 4.4) so
the LTX argument surface can't drift from the upstream script. Each chunk is
run as its own subprocess with ``--reference-video``/``--conditioning-video``
set to the chunk's sliced subvideo (rgb mode) and ``--reference-image`` set to
that chunk's FLUX-edited first frame.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # .../LTX-2/scripts
STYLE_TRANSFER_DIR = SCRIPTS_DIR / "style_transfer_ltx23"
for _dir in (SCRIPTS_DIR, STYLE_TRANSFER_DIR):
    _dir_str = str(_dir)
    if _dir_str not in sys.path:
        sys.path.insert(0, _dir_str)

from style_transfer_preprocess import is_readable_video_file  # noqa: E402
from translate_single_sequence import build_style_transfer_command  # noqa: E402


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    env = os.environ.copy()
    # The 22B IC-LoRA checkpoint runs close to a 32GB GPU's capacity; reduce
    # allocator fragmentation the same way the rest of this repo's IC-LoRA
    # invocations do (e.g. scripts/translate_single_sequence.sh,
    # scripts/detailer/run_detailer_attention_probe_batch.py).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, env=env)


def generate_subvideo(
    ltx_args: argparse.Namespace,
    source_subvideo: Path,
    edited_first_frame: Path,
    num_frames: int,
    out_path: Path,
) -> Path:
    """Run IC-LoRA style transfer for one chunk, writing ``out_path``.

    ``ltx_args`` is the passthrough namespace from ``PipelineConfig.ltx_args``
    (model/checkpoint paths, prompt, strengths, etc.). ``ltx_args.conditioning_mode``
    must be ``"rgb"`` -- the chunk subvideo is used as both the reference video
    and the (identity) conditioning video.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_readable_video_file(out_path):
        print(f"Reusing existing chunk output video: {out_path}")
        return out_path

    if ltx_args.conditioning_mode != "rgb":
        raise ValueError(
            f"chunked ltx_generate stage only supports conditioning_mode='rgb', got "
            f"{ltx_args.conditioning_mode!r}"
        )

    source_subvideo = str(Path(source_subvideo).expanduser().resolve())
    command = build_style_transfer_command(
        ltx_args,
        source_video=source_subvideo,
        control_video=source_subvideo,
        reference_image=str(Path(edited_first_frame).expanduser().resolve()),
        num_frames=num_frames,
        output=str(out_path),
    )
    run_command(command)

    if not out_path.is_file():
        raise FileNotFoundError(f"The IC-LoRA style-transfer runner did not create {out_path}")
    return out_path
