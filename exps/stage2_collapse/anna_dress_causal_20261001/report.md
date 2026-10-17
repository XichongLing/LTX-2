# Anna dress-collapse causal study

Date: 2026-10-01  
Current revision: `055ebb349797e8d2ac25a486c97f1cc14dfd927b` plus the preserved dirty worktree  
Historical revision: `757fc2f`

## Conclusion

The factor that flips the garment phenotype is **generation resolution**, and the divergence is already visible in the clean spatially upscaled Stage-1 latent before Stage-2 denoising.

- At 384x256, the loose edited skirt remains broad and patterned through frame 120 for seeds 42, 43, and 44.
- At 1280x768, the skirt becomes narrow/smooth and source-like by frame 24 for all three seeds.
- The same low/full split is present when the saved clean Stage-1 latents are decoded directly. Therefore the routed Stage-2 loop is not the origin of the difference.
- Prompt wording, negative prompt, 121 versus 129 output frames, historical versus current code, and legacy versus routed-video Stage-2 execution do not flip the phenotype.

The most likely mechanism is that full-resolution Stage 1 has enough spatial capacity for source-video IC-LoRA/tokens to transmit the source garment silhouette and material. The 384x256 path uses an 8x12 upscaled Stage-1 latent grid instead of 24x40: one tenth as many spatial sites per latent frame. That bottleneck suppresses source garment detail and lets the edited first-frame silhouette dominate, but it also removes facial and fabric detail. This mechanism is an inference from the isolated resolution toggle and clean-latent localization; the precise attention-layer pathway was not separately ablated.

## Adaptive run inventory

Ten new generations were made, within the planned 8-12 run budget. Existing historical and low-resolution outputs were reused.

| Stage | Run | Result |
|---|---|---|
| Gate | `commit757_legacy_full_oldcfg` | Collapse; decoded pixels are identical to the historical video (SSIM 1.000000). |
| Gate | `current_legacy_full_oldcfg` | Collapse. |
| Gate | `current_routed_full_oldcfg` | Collapse. |
| Upstream | `upstream_low_historical_prompt` | No collapse. |
| Upstream | `upstream_full_short_prompt` | Collapse. |
| Upstream | `upstream_full_historical_prompt_121` | Collapse. |
| Confirm | `confirm_resolution_seed43_low_short` | No collapse. |
| Confirm | `confirm_resolution_seed43_full_short` | Collapse. |
| Confirm | `confirm_resolution_seed44_low_short` | No collapse. |
| Confirm | `confirm_resolution_seed44_full_short` | Collapse. |

The mechanics factorial was intentionally not launched: both current legacy and routed-video full-resolution gates collapsed, which routed the adaptive study to the upstream configuration branch. The code-drift branch was also unnecessary because both commit `757fc2f` and the current legacy path reproduced collapse.

## Visual checkpoint review

Frames 0, 24, 48, 72, 96, and 120 were reviewed. Values below are the median over post-frame-0 checkpoints from a label-blinded row sheet; resolution itself is visually inferable, so an independent blinded human review would still be appropriate for publication.

| Seed / setting | Loose silhouette (1-5) | Lower pattern (1-5) | Edited upper material (1-5) | Source collapse (1-5) | Collapse time |
|---|---:|---:|---:|---:|---:|
| 42, 384x256 | 5 | 4 | 3 | 1 | none |
| 42, 1280x768 | 1 | 1 | 1 | 5 | 24 |
| 43, 384x256 | 5 | 4 | 3 | 1 | none |
| 43, 1280x768 | 1 | 1 | 1 | 5 | 24 |
| 44, 384x256 | 5 | 4 | 3 | 1 | none |
| 44, 1280x768 | 1 | 1 | 1 | 5 | 24 |

The resolution toggle changes collapse severity by four points and flips all five post-frame-0 checkpoints in the same direction for all three seeds.

The upper body is a separate leakage failure: full-resolution runs rapidly recover the smooth gray/source-like upper garment. Low-resolution runs retain more of the edited material/color but render it poorly. This agrees with the user's observation that the lower dress preserves the first-frame appearance more successfully than the upper garment.

## DINOv2 tracked-region check

The local DINOv2 ViT-B/14 register checkpoint was evaluated on hand-defined upper/lower garment polygons propagated through source motion with dense optical flow. The reported number is the post-frame-0 median

`cos(output, edited first frame) - cos(output, source frame)`.

Positive values favor the edit. This is a secondary metric: approximate optical-flow tracking and crop sensitivity mean absolute values should not be over-interpreted.

| Seed | Upper low | Upper full | Lower low | Lower full |
|---:|---:|---:|---:|---:|
| 42 | 0.0857 | 0.0347 | 0.2086 | 0.0521 |
| 43 | 0.1048 | -0.0029 | 0.1828 | 0.0588 |
| 44 | 0.1803 | 0.0268 | 0.2052 | 0.0143 |

All six paired region comparisons move toward the edited reference at low resolution. The direct Stage-1 decodes show the same direction:

| Clean Stage-1 decode | Upper margin | Lower margin |
|---|---:|---:|
| low, short prompt | 0.1380 | 0.0436 |
| full, short prompt | 0.0240 | -0.0353 |
| low, historical prompt | 0.1218 | 0.0625 |
| full, historical prompt | 0.0100 | -0.0802 |

This localizes the dominant divergence upstream of Stage 2.

## First-step stability

All recorded Gaussian Stage-2 first steps were finite. Norms below are RMS-normalized by the target tensor element count.

| Run family | Input RMS | x0 RMS | Velocity RMS |
|---|---:|---:|---:|
| full gates | 0.9200-0.9204 | 0.9959-1.0015 | 1.3557-1.3586 |
| full resolution confirmations | 0.9202-0.9211 | 1.0051-1.0184 | 1.3579-1.3690 |
| low resolution runs | 0.9256-0.9295 | 1.0255-1.0497 | 1.3645-1.3894 |

There is no catastrophic first-step norm shift that could explain the binary visual change. Exact cache, initialization, and noise checksums are retained in each run's `output.mp4.stage2.json`, `predictions/manifest.json`, and noise-bundle manifest.

## Ranked factors

1. **Output/Stage-1 spatial resolution — causal, high confidence.** It is an isolated toggle, changes the phenotype at all five post-frame-0 checkpoints, repeats for three seeds, shifts both DINO regions consistently, and is visible before Stage 2.
2. **Stage-1 source-conditioning transmission — likely mediator, medium-high confidence.** Full Stage-1 decodes are source-like; low Stage-1 decodes are edit-like. A dedicated Stage-1 source-token/LoRA ablation would be required to separate those two conditioning mechanisms.
3. **Prompt and negative prompt — not causal in this case, high confidence.** Low stays loose and full collapses under both prompt variants.
4. **Frame count (121 versus 129 decoded frames) — not causal, high confidence.** The full historical-prompt outputs collapse in both cases.
5. **Legacy versus routed-video Stage-2 loop — not causal, high confidence.** Both collapse under matched full-resolution historical settings.
6. **Code drift from `757fc2f` to current — not causal, high confidence.** Historical reproduction is pixel-identical and current legacy also collapses.

## Artifacts

- `gate_contact_sheet.jpg`: historical/current and legacy/routed gate.
- `upstream_contact_sheet.jpg`: resolution x prompt and frame-count cells.
- `resolution_confirmation_contact_sheet.jpg`: seed 42/43/44 low/full pairs.
- `stage1_decode_contact_sheet.jpg`: direct decodes of clean upscaled Stage-1 caches.
- `evaluation/dino_margins.png`, `evaluation/dino_margins.csv`, and `evaluation/dino_summary.json`: quantitative curves and values.
- `stage1_decodes/*.mp4`: decoded clean Stage-1 trajectories.
- `*_manifest.json`, per-run `command.json`, `run.log`, `output.mp4.stage2.json`, `predictions/`, cache manifests, and noise-bundle manifests: reproducibility data.

## Practical implication

The non-collapse in `gaussian__source_full` was not evidence that routed Stage 2 weakens source-video conditioning. It was primarily a low-resolution artifact. Keeping that resolution is not a viable final remedy because it trades garment preservation for visibly degraded face and material detail.

The next targeted intervention should therefore happen in **Stage 1 at production resolution**: reduce or spatially mask source-video garment conditioning while retaining motion/pose conditioning, then pass the improved clean latent to unchanged Stage 2. Phi-Noise may help preserve that improved Stage-1 structure, but it cannot recover an edited loose silhouette once the full-resolution Stage-1 latent has already collapsed it.
