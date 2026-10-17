#!/usr/bin/env python3
"""Orchestrator (Section 6.4): plan -> FLUX stage -> LTX stage -> concat.

Stage-batched per Section 4.1: Stage A is CPU-only planning/slicing, Stage B
runs one FLUX subprocess for every chunk's first-frame edit, Stage C runs one
``run_ic_lora_style_transfer.py`` subprocess per chunk, and Stage D
concatenates the per-chunk outputs. Every stage reads/writes ``manifest.json``
and reuses existing artifacts (Section 4.5), so a crash partway through only
re-runs the remaining work.
"""
from __future__ import annotations

from pathlib import Path

from . import io_utils
from .assembly.concat import concat_chunks
from .config import PipelineConfig
from .flux_edit.prompt import build_multiref_prompt
from .flux_edit.runner import FluxEditJob, edit_first_frames_batch
from .ltx_generate.runner import generate_subvideo
from .manifest import Manifest, chunk_dir, read_manifest, write_manifest


def _validate_ltx_checkpoint_paths(cfg: PipelineConfig) -> None:
    """Fail fast on bad checkpoint/LoRA paths, before any GPU work starts.

    A stray character (e.g. a smart quote pasted into --ic-lora-path) would
    otherwise only surface as a quiet "Failed to read metadata from LoRA
    file" warning, with the run proceeding LoRA-less until it OOMs or
    finishes with the wrong style.
    """
    for arg_name in ("distilled_checkpoint_path", "spatial_upsampler_path", "ic_lora_path"):
        path = Path(getattr(cfg.ltx_args, arg_name))
        if not path.is_file():
            flag = "--" + arg_name.replace("_", "-")
            raise FileNotFoundError(f"{flag} does not exist: {path}")

    gemma_root = Path(cfg.ltx_args.gemma_root)
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"--gemma-root does not exist or is not a directory: {gemma_root}")


def plan_and_slice(cfg: PipelineConfig) -> Manifest:
    """Stage A: plan chunks, slice V0, extract each chunk's first frame."""
    if not io_utils.is_readable_video_file(cfg.source_video):
        raise FileNotFoundError(f"source video does not exist or is unreadable: {cfg.source_video}")
    if not cfg.target_reference.is_file():
        raise FileNotFoundError(f"target reference image does not exist: {cfg.target_reference}")

    _width, _height, total_frames, fps = io_utils.read_video_metadata(str(cfg.source_video))
    chunks = cfg.chunker.plan(total_frames, fps)
    if not chunks:
        raise ValueError(f"chunker produced no chunks for a {total_frames}-frame video")

    print(f"Planned {len(chunks)} chunk(s) for {total_frames} frames @ {fps:.3f} fps:")
    for chunk in chunks:
        print(f"  chunk_{chunk.index:03d}: frames {chunk.start_frame}..{chunk.end_frame - 1} ({chunk.num_frames} frames)")

    manifest = Manifest.from_chunks(cfg.source_video, cfg.target_reference, fps, total_frames, chunks)

    slice_paths = io_utils.slice_subvideos(cfg.source_video, chunks, cfg.work_dir)
    for record, source_path in zip(manifest.chunks, slice_paths, strict=True):
        record.source_video = str(source_path)
        first_frame_path = chunk_dir(cfg.work_dir, record) / "first_frame.png"
        io_utils.extract_first_frame_from_video(source_path, first_frame_path)
        record.first_frame = str(first_frame_path)

    write_manifest(cfg.work_dir, manifest)
    return manifest


def flux_edit_all(cfg: PipelineConfig, manifest: Manifest) -> Manifest:
    """Stage B: edit every chunk's first frame toward I0 in one FLUX process."""
    flux_prompt = build_multiref_prompt(cfg.user_prompt, cfg.negative_prompt)

    jobs = [
        FluxEditJob(
            target_reference=cfg.target_reference,
            source_first_frame=Path(record.first_frame),
            output=chunk_dir(cfg.work_dir, record) / "edited_first_frame.png",
            prompt=flux_prompt,
        )
        for record in manifest.chunks
    ]

    edit_first_frames_batch(
        jobs,
        seed=cfg.flux_seed,
        steps=cfg.flux_steps,
        guidance=cfg.flux_guidance,
        device=cfg.flux_device,
        runner_path=cfg.flux_runner_path,
        python=cfg.flux_python,
        model=cfg.flux_model,
        cpu_offload=cfg.flux_cpu_offload,
        height=cfg.flux_height,
        width=cfg.flux_width,
        work_dir=cfg.work_dir,
    )

    for record, job in zip(manifest.chunks, jobs, strict=True):
        record.edited_first_frame = str(job.output)

    write_manifest(cfg.work_dir, manifest)
    return manifest


def ltx_generate_all(cfg: PipelineConfig, manifest: Manifest) -> Manifest:
    """Stage C: run IC-LoRA style transfer for every chunk."""
    for record in manifest.chunks:
        out_path = chunk_dir(cfg.work_dir, record) / "output.mp4"
        print(f"=== Chunk {record.index}: frames {record.start_frame}..{record.start_frame + record.num_frames - 1} ===")
        generate_subvideo(
            cfg.ltx_args,
            source_subvideo=record.source_video,
            edited_first_frame=record.edited_first_frame,
            num_frames=record.num_frames,
            out_path=out_path,
        )
        record.output_video = str(out_path)
        write_manifest(cfg.work_dir, manifest)

    return manifest


def assemble(cfg: PipelineConfig, manifest: Manifest) -> Path:
    """Stage D: concatenate the per-chunk outputs into the final video.

    Chunks whose plan shifted their start backward to cover the full source
    span (Section 4.3) overlap the previous chunk by ``overlap_in`` frames;
    trim that overlap from each such chunk's output before concatenating so
    the final video's frame count matches the source's.
    """
    chunk_outputs = []
    for record in manifest.chunks:
        output_video = Path(record.output_video)
        if record.overlap_in > 0:
            trimmed = chunk_dir(cfg.work_dir, record) / "output_trimmed.mp4"
            output_video = io_utils.trim_overlap(output_video, record.overlap_in, record.num_frames, trimmed, manifest.fps)
        chunk_outputs.append(output_video)
    return concat_chunks(chunk_outputs, cfg.output, manifest.fps)


def run(cfg: PipelineConfig) -> Path:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    _validate_ltx_checkpoint_paths(cfg)

    manifest = read_manifest(cfg.work_dir)
    if manifest is None:
        manifest = plan_and_slice(cfg)
    else:
        print(f"Resuming from existing manifest with {len(manifest.chunks)} chunk(s): {cfg.work_dir / 'manifest.json'}")

    manifest = flux_edit_all(cfg, manifest)
    manifest = ltx_generate_all(cfg, manifest)
    output_path = assemble(cfg, manifest)

    print(f"Chunked style transfer completed successfully: {output_path}")
    return output_path
