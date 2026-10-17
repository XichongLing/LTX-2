# Stage-1 source-correspondence bias

## Goal

Increase the pre-softmax self-attention logits from masked Stage-1 target
queries to their aligned source-video keys. Source K/V tensors remain unchanged
and already exist in the IC-LoRA token sequence.

## Design choices

- **Stage 1 only.** The observed artifact originates in Stage 1, so Stage 2 is
  deliberately unchanged.
- **Directional bias.** Only target-query to source-key weights are modified.
  Source-to-target, target-to-target, image-conditioning, and source self
  attention retain their previous values.
- **Reuse attention weights.** LTX already converts positive self-attention
  weights with `log(weight)` before SDPA. Storing `exp(mask * logit_bias)`
  therefore produces the exact additive pre-softmax bias without adding a new
  state field or touching K/V.
- **First video condition is the source.** The style-transfer runner orders RGB
  first in multi-condition modes. Depth and edge reference groups do not receive
  the correspondence boost.
- **Normalized latent correspondence.** Target `(t,h,w)` coordinates map to the
  source grid by normalized floor mapping. This supports IC-LoRAs whose source
  reference is spatially downscaled.
- **Canonical mask contract.** Every mask provider emits float
  `(1,1,F,H,W)` values in `[0,1]`. Attention code is independent of whether the
  mask came from a procedural rule or a file.
- **Causal temporal pooling.** Pixel frame 0 maps to latent frame 0. Remaining
  pixel frames are grouped according to the causal VAE layout and max-pooled so
  a brief disocclusion is not averaged away.
- **Dense prototype.** Enabling the feature materializes a dense attention
  weight matrix and selects a masked attention backend. Sparse/kernel
  optimization is intentionally deferred until visual validation.

## Mask sources

- `later-frames`: zero on frame 0 and one everywhere on later frames. Requires
  no external mask and is intended as an end-to-end mechanism test.
- `boxes`: repeatable frame-inclusive boxes with normalized coordinates.
- `file`: video, single image, image directory, `.npy`, or `.pt`. A single
  image is broadcast across time. File masks use the same resize-and-center-crop
  operation as pipeline media.

## Modified files

- `packages/ltx-core/src/ltx_core/conditioning/types/correspondence_attention_bias_wrapper.py`
  - Added directional correspondence mapping and conditioning wrapper.
- `packages/ltx-core/src/ltx_core/conditioning/types/__init__.py`
  - Exported the new wrapper.
- `packages/ltx-core/src/ltx_core/conditioning/__init__.py`
  - Exported the new wrapper from the public conditioning package.
- `packages/ltx-core/src/ltx_core/types.py`
  - Documented that positive attention weights may exceed one to express boosts.
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
  - Updated attention-weight semantics.
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
  - Updated log-space conversion documentation.
- `packages/ltx-pipelines/src/ltx_pipelines/correspondence_mask.py`
  - Added procedural/file mask providers and target-latent conversion.
- `packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`
  - Added Stage-1 API plumbing and wrapped only the first video condition.
  - Replaced a stale missing `_downsample_mask_to_latent` call with the shared
    `downsample_mask_video_to_latent` helper.
- `scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py`
- `scripts/style_transfer_ltx23/translate_single_sequence.py`
  - Exposed and forwarded the options through the launcher used by existing
    experiment scripts.
  - Added CLI selection, validation, mask construction, pipeline arguments, and
    attention-probe metadata.
- `tests/test_correspondence_attention_bias.py`
  - Added tests for causal/max temporal pooling, box masks, directional gating,
    existing-weight composition, and neighborhood radius.

## CLI examples

No external mask:

```bash
--source-correspondence-bias 0.405465
--correspondence-mask-mode later-frames
```

Approximate region:

```bash
--source-correspondence-bias 1.386294
--source-correspondence-radius 1
--correspondence-mask-mode boxes
--correspondence-mask-box 24 56 0.65 0.15 0.98 0.85
--correspondence-mask-feather 8
```

Manual mask sequence:

```bash
--source-correspondence-bias 1.386294
--correspondence-mask-mode file
--correspondence-mask-file /path/to/grayscale_png_directory
```

## Suggested initial sweep

For the broad `later-frames` mask, use weaker logits:

- `log(1.25) = 0.223144`
- `log(1.5) = 0.405465`
- `log(2) = 0.693147`

For a localized box or manual mask, also test `log(4) = 1.386294`.
