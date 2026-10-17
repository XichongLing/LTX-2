# Stage 2 IC-LoRA implementation record

Date: 2026-08-04

## Goal

Allow the full-resolution Stage 2 diffusion transformer to use the same IC-LoRA
mechanism as Stage 1. This includes both parts of IC-LoRA inference:

1. Fuse IC-LoRA weight deltas into the Stage 2 transformer.
2. Append encoded source-video reference tokens to the Stage 2 video sequence.

The previous behavior remains the default.

## Modified files

### `packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`

- Added the optional `stage_2_loras` constructor argument.
- Stage 2 now receives those adapters instead of always using `loras=()`.
- Added `stage_2_ic_lora_enabled` as an explicit pipeline state flag.
- Reads and validates reference-downscale metadata across both Stage 1 and
  Stage 2 adapters.
- Added `stage_2_conditioning_attention_strength` to the generation call.
- When Stage 2 IC-LoRA is enabled, Stage 2 calls `_create_conditionings()` at
  the full target resolution. This preserves image conditionings and appends
  newly encoded source-video reference tokens.
- When it is disabled, Stage 2 retains the previous
  `combined_image_conditionings()` path.
- Added validation for Stage 2 attention strength and for missing source-video
  conditionings.
- Added an execution log identifying the number of Stage 2 source conditions
  and their attention strength.

### `scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py`

- Added `--stage-2-ic-lora-strength`, default `0.0`.
- Added `--stage-2-conditioning-attention-strength`, default `1.0`.
- Constructs a separate Stage 2 adapter descriptor so Stage 1 remains fixed at
  its existing adapter strength.
- A positive Stage 2 adapter strength enables both adapter fusion and Stage 2
  reference-video token insertion.
- Added argument validation, terminal summaries, and attention-probe metadata.

### `scripts/style_transfer_ltx23/translate_single_sequence.py`

- Added both Stage 2 arguments to the wrapper CLI.
- Forwards both values to `run_ic_lora_style_transfer.py`.

## Design choices

### Backward-compatible zero setting

`--stage-2-ic-lora-strength 0.0` creates no Stage 2 adapter descriptor. Stage 2
therefore follows its historical image-only conditioning branch. Existing
commands produce the previous architecture unless the new option is supplied.

### Adapter and reference tokens are enabled together

The IC-LoRA adapter was trained with reference and target tokens concatenated in
one transformer sequence. Enabling only the adapter does not reproduce that
training setup. Consequently, a positive Stage 2 adapter strength also appends
the source-video reference tokens.

### Separate adapter and attention controls

`--stage-2-ic-lora-strength` scales the learned LoRA weight deltas.

`--stage-2-conditioning-attention-strength` controls access to the appended
reference tokens. These are separate mechanisms and should not share a value.

The existing `--stage-2-masked-denoise-strength` remains independent: it controls
how much noise and denoising the upsampled Stage 1 target tokens receive.

### Source video is encoded again for Stage 2

Stage 2 has a different target spatial resolution, so Stage 1 reference latents
cannot be reused directly. `_create_conditionings()` loads and VAE-encodes the
source at the Stage 2 reference resolution. For the current adapter metadata,
`reference_downscale_factor=2`, so a Stage 2 target of `H x W` uses reference
latents encoded from `H/2 x W/2` pixels.

### No Stage 2 correspondence bias yet

The custom target-to-source correspondence bias remains Stage 1-only. The first
Stage 2 experiments should isolate standard IC-LoRA behavior. Stage 2-specific
correspondence routing can be added later under separate controls.

## Validation performed

- Python compilation of all three modified Python files.
- Runner `--help` startup and visibility of both new CLI options.
- Ruff was run on the modified files. The new focused test passes Ruff; the
  existing pipeline and runner files retain pre-existing repository warnings.
- Focused unit tests for Stage 2 adapter configuration and wrapper forwarding.

## Suggested first experiment

Use identical inputs and seed for:

1. `--stage-2-ic-lora-strength 0.0`
2. `--stage-2-ic-lora-strength 0.25`
3. `--stage-2-ic-lora-strength 0.5`
4. `--stage-2-ic-lora-strength 1.0`

Keep `--stage-2-conditioning-attention-strength 1.0` and
`--stage-2-masked-denoise-strength 1.0` fixed for the first sweep. This isolates
the effect of adding IC-LoRA to Stage 2.

## Expected cost

With `reference_downscale_factor=2`, reference video tokens add approximately
one quarter of the Stage 2 target spatial-token count. The joint attention
sequence is therefore about 1.25 times as long, corresponding to roughly 1.56
times the self-attention matrix work before accounting for other modalities.
The main expected cost is Stage 2 attention memory and runtime rather than
persistent adapter storage.
