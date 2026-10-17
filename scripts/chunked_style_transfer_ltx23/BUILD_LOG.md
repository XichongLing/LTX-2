# Build Log: Chunked Reference-Guided Style Transfer

Implements `chunked_style_transfer_plan.md`: a chunk-loop wrapper around the
existing `scripts/style_transfer_ltx23/` IC-LoRA pipeline that re-grounds
each temporal chunk with a FLUX.2 two-reference edit of its first frame
before running IC-LoRA on it.

All paths below are relative to `/home/xichong-drz-inokko/experimental/LTX-2/`
unless noted otherwise.

## New package: `scripts/chunked_style_transfer_ltx23/`

| File | Purpose |
| --- | --- |
| `__init__.py` | Empty package marker. |
| `chunking/__init__.py` | Re-exports `ChunkSpec`, `VideoChunker`, `UniformIntervalChunker`. |
| `chunking/base.py` | `ChunkSpec` (frozen dataclass: `index`, `start_frame`, `num_frames`, `overlap_in`, with `end_frame`/`dir_name` properties) and the `VideoChunker` ABC (`plan(total_frames, fps) -> list[ChunkSpec]`). |
| `chunking/uniform.py` | `snap_down(value, lcm_constraint, base_offset)` and `UniformIntervalChunker(target_len, lcm_constraint=8, base_offset=1, min_chunk_frames=None)`. Emits fixed-length chunks snapped to LTX-valid lengths (`(num_frames-1) % 8 == 0`), merges a too-short trailing remainder into the previous chunk. |
| `io_utils.py` | Adds `style_transfer_ltx23/` to `sys.path` and re-exports `extract_first_frame_from_video`, `extract_frame_from_video`, `is_readable_video_file`, `read_video_metadata` from `style_transfer_preprocess`. Adds `slice_subvideos(source_video, chunks, work_dir)`, a thin per-chunk loop over `translate_single_sequence_chunked.slice_video_by_frames`. |
| `manifest.py` | `ChunkRecord` (per-chunk artifact paths: `source_video`, `first_frame`, `edited_first_frame`, `output_video`) and `Manifest` (source/target/fps/total_frames + `chunks: list[ChunkRecord]`), with `to_json`/`from_json`, `write_manifest`/`read_manifest`, `manifest_path`, `chunk_dir`. |
| `config.py` | `PipelineConfig` dataclass + `from_args(args)`. Builds the `UniformIntervalChunker`, resolves `DEFAULT_FLUX_MULTIREF_RUNNER` (sibling repo path, see below), and builds the `ltx_args` namespace forwarded to `build_style_transfer_command`, pinning `ltx_args.conditioning_mode = "rgb"`. |
| `flux_edit/__init__.py` | Empty. |
| `flux_edit/prompt.py` | `MULTIREF_PREAMBLE` + `build_multiref_prompt(user_prompt, negative_prompt="")`, disambiguating "image 1 = style target I0" vs "image 2 = scene to restyle" for the FLUX runner. |
| `flux_edit/runner.py` | `FluxEditJob` dataclass; `edit_first_frames_batch(jobs, ..., runner_path, work_dir)` writes `work_dir/flux_jobs.json` and invokes `run_flux_multiref_edit.py --jobs-json ...` **once** for all chunks (skips the subprocess entirely if every output already exists); `edit_first_frame(...)` is a single-job convenience wrapper around the batch path. |
| `ltx_generate/__init__.py` | Empty. |
| `ltx_generate/runner.py` | Adds `style_transfer_ltx23/` to `sys.path`; `generate_subvideo(ltx_args, source_subvideo, edited_first_frame, num_frames, out_path)` reuses `translate_single_sequence.build_style_transfer_command` with `source_video=control_video=<chunk subvideo>` and `reference_image=<FLUX-edited first frame>`, runs it via subprocess. Reuse-if-exists via `is_readable_video_file(out_path)`. Asserts `ltx_args.conditioning_mode == "rgb"`. |
| `assembly/__init__.py` | Empty. |
| `assembly/concat.py` | `concat_chunks(chunk_outputs, output_path, fps)` reuses `translate_single_sequence_chunked.concatenate_videos`. Reuse-if-exists via `is_readable_video_file(output_path)`. Naive non-overlapping concat (no seam blending yet). |
| `pipeline.py` | Orchestrator: `plan_and_slice` (Stage A: metadata -> chunk plan -> manifest -> slice subvideos -> extract first frames), `flux_edit_all` (Stage B: one FLUX subprocess for all chunks), `ltx_generate_all` (Stage C: one IC-LoRA subprocess per chunk, manifest written after each chunk), `assemble` (Stage D: concat), `run` (top-level; resumes from an existing `manifest.json` if present). |
| `cli.py` | Argparse entry point. Core I/O (`--source-video`, `--target-reference`, `--work-dir`, `--output`, `--prompt`, `--negative-prompt`), chunking (`--chunk-len`, `--lcm-constraint`, `--base-offset`, `--divider`), FLUX passthrough (`--flux-*`), and LTX/IC-LoRA passthrough mirroring `translate_single_sequence.py`'s arguments (checkpoint paths, strengths, attention-probe options, `--quantization`, etc.). `main()` builds a `PipelineConfig` and calls `run()`. |

## New entry point

- `scripts/run_chunked_style_transfer.py` — adds `scripts/` to `sys.path` and
  calls `chunked_style_transfer_ltx23.cli.main()`.

## New sibling script (different repo)

- `/home/xichong-drz-inokko/experimental/flux.2_dev/run_flux_multiref_edit.py`
  (**not** under `LTX-2/`) — two(+)-reference FLUX.2-dev edit runner, sibling
  to `run_flux_first_frame_style_transfer.py`. Two modes:
  - Single job: `--reference A --reference B --output OUT --prompt "..."`
  - Batch: `--jobs-json JOBS.json` where each entry is
    `{"references": [path, ...], "output": path, "prompt": str}`. Loads the
    FLUX pipeline **once** and processes every job whose output doesn't
    already exist (this is what `flux_edit/runner.py` uses for Stage B).

  Output `height`/`width` default to the *last* reference image's dimensions
  rounded to a multiple of 16 (the chunk's source first frame, "image 2"),
  unless overridden by `--height`/`--width`. Appends a structure-preservation
  suffix (`STRUCTURE_GUARDRAILS_SUFFIX`) to the prompt by default
  (`--no-append-structure-guardrails` to disable).

## Key design decisions

- **LTX frame-count rule (Section 4.2)**: valid chunk lengths satisfy
  `(num_frames - 1) % 8 == 0`, confirmed from `VIDEO_SCALE_FACTORS.time` in
  `packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py`. Encoded as
  `UniformIntervalChunker`'s defaults `lcm_constraint=8, base_offset=1`.

- **Reuse by import, not copy (Section 4.4)**:
  - `extract_frame_from_video`, `extract_first_frame_from_video`,
    `read_video_metadata`, `is_readable_video_file` ← `style_transfer_preprocess`
  - `slice_video_by_frames`, `concatenate_videos` ← `translate_single_sequence_chunked`
    (discovered mid-build; these already implement frame-accurate slicing and
    sequential-pts concatenation with reuse-if-exists, so `io_utils.py` and
    `assembly/concat.py` are thin wrappers rather than reimplementations).
  - `build_style_transfer_command` ← `translate_single_sequence` — used
    as-is by `ltx_generate/runner.py` so the IC-LoRA CLI surface can't drift
    from upstream.

- **Stage-batched execution (Section 4.1)**:
  - Stage A (planning/slicing/first-frame extraction) is CPU-only.
  - Stage B (FLUX edits) is fully batched: one subprocess, one model load,
    all N chunks, via `run_flux_multiref_edit.py --jobs-json`.
  - Stage C (IC-LoRA) runs `run_ic_lora_style_transfer.py` once **per chunk**
    via subprocess — this script was *not* modified to add native batch
    support, since it's a core GPU inference script that can't be validated
    without hardware/checkpoints. This matches the existing precedent in
    `translate_single_sequence_chunked.py` (itself a per-chunk wrapper).
    **Follow-up**: a possible future "Phase 2.5" could add an optional
    `--batch-jobs-json`-style mode to `run_ic_lora_style_transfer.py` so
    Stage C also amortizes one model load across all chunks, fully
    satisfying 4.1 end-to-end.
  - Stage D (concat) is CPU-only.

- **Manifest-driven, resumable (Section 4.5)**: `manifest.json` in
  `--work-dir` records the chunk plan and per-chunk artifact paths. `run()`
  resumes from an existing manifest if present; each stage/chunk also does
  its own file-existence reuse check (`is_readable_video_file`, output image
  exists, etc.), so a crash partway through only re-runs missing work.

- **`conditioning_mode` pinned to `"rgb"`**: `PipelineConfig.from_args` forces
  `ltx_args.conditioning_mode = "rgb"` because Stage C always feeds the same
  chunk subvideo as both the reference video and the (identity) conditioning
  video.

## Post-build fix: `--flux-python` default

First real run failed with `ModuleNotFoundError: No module named 'diffusers'`
because Stage B's subprocess defaulted to `sys.executable` (the LTX-2 venv,
which doesn't have `diffusers`). `flux.2_dev` has its own venv at
`/home/xichong-drz-inokko/experimental/.venv` (the one
`flux.2_dev/run_low_fidelity_fixed_photoreal.sh` already hardcodes as
`FLUX_PYTHON`), which has `diffusers`/`transformers`/`torch` with CUDA.

Fix: `config.py` now defines `DEFAULT_FLUX_PYTHON = REPO_ROOT.parent / ".venv" / "bin" / "python"`.
`PipelineConfig.from_args` uses this as the `flux_python` default when
`--flux-python` isn't passed and the path exists; otherwise it falls back to
`sys.executable` as before. `--flux-python` can still override it.

## Post-build fix: Stage C OOM / bad checkpoint paths

A real Stage C run hit `torch.OutOfMemoryError` loading the 22B IC-LoRA
checkpoint (29.6/31.36 GiB used, failed allocating +128 MiB), and separately
the run's `--ic-lora-path` contained stray smart-quote characters
(`”...“`), which `run_ic_lora_style_transfer.py` only surfaced as a quiet
"Failed to read metadata from LoRA file" warning — the run proceeded
LoRA-less and then OOM'd minutes later.

Two fixes:

- `ltx_generate/runner.py`'s `run_command` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `PYTORCH_ALLOC_CONF=expandable_segments:True` (via `env.setdefault`, so an
  existing env value wins) on the IC-LoRA subprocess, matching the convention
  used throughout this repo's other IC-LoRA invocations
  (`scripts/translate_single_sequence.sh`,
  `scripts/detailer/run_detailer_attention_probe_batch.py`).
- `pipeline.py` adds `_validate_ltx_checkpoint_paths(cfg)`, called at the top
  of `run()`: checks `--distilled-checkpoint-path`, `--spatial-upsampler-path`,
  `--ic-lora-path` are files and `--gemma-root` is a directory, raising
  `FileNotFoundError` immediately if not — instead of failing only after a
  slow GPU model load.

For the 22B checkpoint on a 32GB GPU, also pass `--quantization fp8-cast`
(and consider `--quiet-layer-streaming` / `--skip-stage-2`), per
`scripts/translate_single_sequence.sh`'s recommended invocation for large
checkpoints.

## Post-build fix: chunk frame-count mismatch (dropped trailing frames)

A real run on a 120-frame source with `--chunk-len 60` produced two 57-frame
chunks (114 frames total) — 6 source frames short. Root cause:
`UniformIntervalChunker.plan()`'s remainder-merge path used `snap_down` for
both the regular chunk length AND the trailing remainder, so the last chunk
was silently shortened to the same `chunk_len` and the final `total_frames -
2*chunk_len` frames were never sliced into any chunk.

Fix, in `chunking/uniform.py`:

- Added `snap_up(value, lcm_constraint, base_offset)` (smallest valid length
  `>= value`), alongside the existing `snap_down`.
- `plan()`'s remainder handling now snaps the trailing span UP. If that
  overshoots `total_frames`, the chunk's `start_frame` is shifted backward by
  the overshoot so it still ends exactly at `total_frames` — guaranteeing
  every chunk together covers `[0, total_frames)`. The backward shift amount
  is recorded as `overlap_in` on that chunk's `ChunkSpec`/`ChunkRecord` (it
  duplicates frames already produced by the previous chunk). The only
  remaining frame-loss case is a single chunk shorter than `base_offset`
  frames total, with no earlier frames to borrow from — unchanged from
  before and unavoidable given the LTX length constraint.
- `pipeline.assemble()` now trims each chunk's leading `overlap_in` frames
  off its `output.mp4` (via the new `io_utils.trim_overlap`) before handing
  the list to `concat_chunks`, so the final video's frame count equals the
  source's.

While testing `trim_overlap` against a real chunk `output.mp4`, found a
second, independent bug: `run_ic_lora_style_transfer.py` writes per-chunk
`output.mp4` files with bogus video-stream rate metadata — observed
`avg_frame_rate=1368/1` (= `57 frames * 24fps`) and `time_base=1/10944` on a
57-frame, 24fps chunk, instead of `24/1`. `slice_video_by_frames` (in
`translate_single_sequence_chunked.py`, reused for `trim_overlap`) derived
its output `time_base` from this bogus `average_rate`, which made libx264's
B-frame reordering round multiple frames' pts to the same dts, producing
`av.error.ValueError: [Errno 22] Invalid argument` on mux (non-monotonic
DTS). `concatenate_videos` was unaffected because it already ignores input
timestamps and assigns its own sequential `pts`/`time_base` from `fps`.

Fix: `slice_video_by_frames` gained an optional `rate: int | Fraction | None`
parameter (default `None` preserves old behavior for its other caller,
`io_utils.slice_subvideos`, slicing the source video which has correct
metadata) and now assigns sequential `pts`/`time_base` from `rate` the same
way `concatenate_videos` does, instead of relying on the input stream's
`average_rate`/auto-pts. `io_utils.trim_overlap` gained a required `fps`
argument and calls `slice_video_by_frames(..., rate=int(round(fps)))`,
overriding the chunk output's bogus metadata with the manifest's real fps.
`pipeline.assemble()` passes `manifest.fps`.

**Note for existing work-dirs**: a `manifest.json` written before this fix
(and its `chunks/*/output.mp4`) reflects the old, frame-dropping chunk plan
(no `overlap_in`). Delete `manifest.json` and `chunks/` (or use a fresh
`--work-dir`) so the corrected plan is regenerated on the next run.

Verified:

- `UniformIntervalChunker.plan()` unit-tested for `(total_frames,
  target_len)` in `(120, 60)`, `(242, 121)`, `(300, 121)`, `(245, 121)`,
  `(122, 61)`, `(50, 121)`, `(5, 121)`, `(1, 121)` — all cover
  `[0, total_frames)` except the two single-chunk-shorter-than-`base_offset`
  edge cases (`50`/`5` frames with `target_len=121`), which lose `<
  lcm_constraint` trailing frames as before.
- `io_utils.trim_overlap("chunks/chunk_001/output.mp4", overlap_in=2,
  num_frames=57, out_path=..., fps=24.0)` against a real chunk output:
  produces a 55-frame, `24/1` fps clip (was: `av.error.ValueError: [Errno 22]
  Invalid argument` before the fix).
- `concat_chunks([chunk_000/output.mp4, <trimmed chunk_001>], ..., fps=24.0)`:
  57 + 55 = 112 frames, `24/1` fps, correct duration — confirms the
  trim+concat path produces a frame-accurate result.

## Testing performed

- `python -m py_compile` on every new module — all pass.
- `python scripts/run_chunked_style_transfer.py --help` — prints the full
  CLI usage correctly.
- Unit-tested `UniformIntervalChunker.plan()` against total frame counts of
  242, 300, 245, and 50 with `target_len=121` (and similar): all emitted
  chunks satisfy `(num_frames-1) % 8 == 0`, and the too-short-remainder merge
  path was exercised.
- End-to-end test of Stage A (`plan_and_slice`) against a real video
  (`outputs/results/fox1.mp4`, 1280x704, 145 frames, 30 fps) with
  `target_len=33`: produced chunks of `[33, 33, 33, 33, 9]` frames, sliced
  each correctly (frame counts verified), extracted first frames for each
  chunk, and round-tripped the manifest through `write_manifest`/`read_manifest`.
- Verified `build_multiref_prompt`, `PipelineConfig.from_args` (full CLI ->
  config, including `flux_runner_path` resolution and `conditioning_mode`
  pinning), and that `ltx_generate.runner` / `assembly.concat` import cleanly
  alongside `translate_single_sequence`/`translate_single_sequence_chunked`.

Stages B (FLUX multi-reference edit) and C (IC-LoRA per chunk) require a GPU
and model checkpoints and have **not** been run end-to-end; they follow the
same subprocess/CLI patterns as the existing single-reference FLUX runner and
`translate_single_sequence_chunked.py` respectively.

## Example usage

```bash
python scripts/run_chunked_style_transfer.py \
  --source-video V0.mp4 \
  --target-reference I0.png \
  --work-dir runs/exp1 \
  --output runs/exp1/final.mp4 \
  --chunk-len 121 \
  --prompt "Restyle as a watercolor painting" \
  --distilled-checkpoint-path /path/to/distilled.safetensors \
  --spatial-upsampler-path /path/to/upsampler.safetensors \
  --gemma-root /path/to/gemma \
  --ic-lora-path /path/to/ic_lora.safetensors
```
