# Stage-2 prediction routing

Controlled Stage 2 uses one FP8/BF16 base transformer with a switchable runtime IC-LoRA. The image and video branches see the same target-video latent; only the video branch receives reference-video tokens and a nonzero adapter scale. Global and spatial modes evaluate both branches at every Stage-2 denoising step.

## Modes

- `legacy`: existing fused Stage-2 behavior; this remains the default.
- `image`: image-conditioned base branch only.
- `video`: image + source-video + runtime IC-LoRA branch only.
- `global`: `(1-g) * image_x0 + g * video_x0`.
- `spatial`: full video prediction outside the mask and `--stage-2-dress-video-contribution` inside it. Mask value `1` means dress.

Controlled modes support unquantized BF16 and `--quantization fp8-cast`. They intentionally reject compile, layer-streaming, other quantization policies, and attention probing.

## Reproducible baseline workflow

First run Stage 1 once and cache its upsampled video/audio inputs:

```bash
uv run python scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py \
  <common arguments> \
  --stage-2-branch-mode image \
  --stage-2-ic-lora-strength 1 \
  --stage-2-noise-seed 42 \
  --save-stage-2-input artifacts/stage2_input.safetensors
```

Then reuse the cache for `video`, `global`, and `spatial` runs:

```bash
uv run python scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py \
  <the same common arguments> \
  --stage-2-branch-mode global \
  --stage-2-video-mix 0.5 \
  --stage-2-ic-lora-strength 1 \
  --stage-2-noise-seed 42 \
  --load-stage-2-input artifacts/stage2_input.safetensors \
  --stage-2-prediction-dir artifacts/predictions/global
```

A spatial dress run additionally uses:

```bash
--stage-2-branch-mode spatial \
--stage-2-routing-mask masks/dress_frames \
--stage-2-dress-video-contribution 0.25
```

The mask may be an image, video, image directory, `.npy`, or tensor file. It must already be dilated, feathered, and temporally smoothed. Inference applies the same resize-and-center-crop geometry as the pipeline, then spatial area and causal temporal mean pooling.

Each controlled output gets an `.stage2.json` sidecar. Prediction recording writes one safetensors file per denoising step plus a manifest containing the input checksum and branch-difference metrics. The Stage-2 cache has its own JSON manifest and refuses reuse when the Stage-1 prompt, seed, dimensions, input asset signatures, or conditioning schedules differ.
