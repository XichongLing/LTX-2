# LTX-2 ContextFlow Attention Audit

Date: 2026-07-13
LTX-2 commit: `6039e16435c330eac90ed64c60badb04990884cc`
ContextFlow commit: `a9b903edf4a3583bd8ef3d71b42ca3b53fbfba71`

## Pipeline Entry Point

- Local LTX-2 pipeline building blocks live under `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`.
- The reusable transformer lifecycle block is `DiffusionStage`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:192`
- The denoiser API consumed by `DiffusionStage` is defined in `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py:58`

## Denoising Loop

- Default loop: `euler_denoising_loop`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:29`
- Available second-order loop: `res2s_audio_video_denoising_loop`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:206`
- Default first-order stepper: `EulerDiffusionStep`.
  - Evidence: `packages/ltx-core/src/ltx_core/components/diffusion_steps.py:22`
- Available second-order stepper: `Res2sDiffusionStep`.
  - Evidence: `packages/ltx-core/src/ltx_core/components/diffusion_steps.py:43`

## Transformer Invocation

- `DiffusionStage.run` builds `LatentState`, wraps the transformer with `BatchSplitAdapter`, and calls the chosen denoising loop.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:318`
- `SimpleDenoiser` and `GuidedDenoiser` convert `LatentState` to `Modality` with `modality_from_latent_state`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:208`
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:262`
- `X0Model.forward` delegates to `LTXModel.forward`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/model.py:495`

## Transformer Forward Path

- `LTXModel.forward` preprocesses `Modality` into `TransformerArgs`, then iterates `self.transformer_blocks`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/model.py:412`
- Block loop: `LTXModel._process_transformer_blocks`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/model.py:347`
- Per-block module: `BasicAVTransformerBlock`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/transformer.py:80`

## Video Self-Attention Class

- Video self-attention module inside each transformer block: `BasicAVTransformerBlock.attn1`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/transformer.py:102`
- The concrete attention implementation is `Attention`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:443`

## Q/K/V Shapes

- `Attention.forward` expects `x` shaped `(B, T, query_dim)` and optional `context` shaped `(B, S, context_dim)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:500`
- `to_q`, `to_k`, and `to_v` each project to `inner_dim = heads * dim_head`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:467`
- Backend call signatures use flattened head dimension before backend-specific reshaping.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`

## Q/K Normalization And RoPE Order

- `Attention.forward` computes `q = self.to_q(x)` and `k = self.to_k(context)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`
- `PytorchPreAttention` applies `q_norm`, then `k_norm`, then RoPE to both `q` and `k`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/ops.py:22`
- `v` is projected before the Q/K preattention path and is not RoPE-transformed.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:524`

Resolution:
LTX-2 applies Q/K normalization before RoPE and immediately calls the attention backend afterward.

Decision:
Capture source K/V immediately before the backend call, after Q/K normalization and RoPE, while preserving the original V tensor.

Confidence:
Directly verified.

## Attention Backend

- Unmasked path: `attention_function(q, k, v, self.heads)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:532`
- Masked path: `masked_attention_function(q, k, v, self.heads, mask)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:534`
- Default backend selection is dynamic SDPA/xFormers/FlashAttention via `AttentionOps`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:432`

## Sequence Dimension

- Token dimension entering `Attention.forward` is dimension `1` in `(B, T, D)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:500`
- After projection, Q/K/V remain `(B, T, H*D_head)` until backend wrappers reshape them.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`

Decision:
For LTX-2, ContextFlow should concatenate source and target K/V on tensor dimension `1`.

Confidence:
Directly verified.

## Block Count

- Transformer blocks are stored in `self.transformer_blocks` and created from `num_layers`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/model.py:324`

Uncertainty:
The exact block count depends on the checkpoint/model configuration used to build `LTXModel`.

Required check:
Read the built transformer config or loaded checkpoint metadata during a real run.

Implementation status:
Not blocked for hook integration because block indices are assigned by loop order in `_process_transformer_blocks`.

## Guidance Batch Layout

- Guided inference batches multiple passes into one transformer call in `_guided_denoise`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:52`
- Pass order is `cond`, optional `uncond`, optional `ptb`, optional `mod`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:96`
- The batch is created by concatenating contexts and repeating latent states.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py:139`

Decision:
ContextFlow should avoid piggybacking on guidance batch ordering for source/target branch identity. Use explicit branch metadata instead.

Confidence:
Directly verified.

## Source-Video Conditioning Path

- `VideoConditionByLatentIndex` replaces target latent tokens in place and optionally adjusts attention strength toward first-frame tokens.
  - Evidence: `packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py:11`
- `VideoConditionByKeyframeIndex` appends guiding keyframe tokens and extends the self-attention mask.
  - Evidence: `packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py:10`

Decision:
Use native LTX-2 conditioning items for source first-frame and target edited-first-frame constraints rather than adding a parallel conditioning path.

Confidence:
Directly verified.

## Edited-First-Frame Conditioning Path

- `modality_from_latent_state` forwards `LatentState.attention_mask` into `Modality`.
  - Evidence: `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:262`
- `TransformerArgsPreprocessor._prepare_self_attention_mask` converts a `(B, T, T)` mask in `[0, 1]` into additive log-space bias with output shape `(B, 1, T_q, T_k)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py:147`

Decision:
Any ContextFlow K/V concatenation must extend the already-prepared additive self-attention bias, not the raw `[0, 1]` mask format.

Confidence:
Directly verified.

## Proposed Hook Location

- Current video self-attention call site:
  - `BasicAVTransformerBlock.forward` -> `self.attn1(...)`
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/transformer.py:269`
- Current backend boundary:
  - `Attention.forward` after `q, k = self.preattention_function(...)` and before `attention_function(...)`.
  - Evidence: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:531`

Proposed hook:

1. Thread explicit branch and sampling-step metadata from `Modality`/`TransformerArgs` into the video self-attention call.
2. In `Attention.forward`, when running video self-attention with a controller:
   - record source `(k, v)` after Q/K normalization + RoPE;
   - retrieve and concatenate source K/V for target;
   - extend the additive self-attention bias for the new key length;
   - preserve the original backend call otherwise.

## ContextFlow-To-LTX-2 Mapping Table

| ContextFlow operation | ContextFlow code location | LTX-2 integration point | Mapping confidence | Notes |
| --- | --- | --- | --- | --- |
| Source K/V capture | `wan/modules/model.py:127`, `wan/modules/model.py:152` | `ltx_core/model/transformer/attention.py:531` | Adapted | Upstream repo is a PnP batch trick, but it confirms early-step, selected-layer intervention during self-attention. |
| K/V concatenation | Paper Section 3.4, Figure 3 | `ltx_core/model/transformer/attention.py:531` | Adapted | Implement as explicit source-target K/V concatenation on token dimension 1. |
| Active step window | `generate.py:557-562`, README `tau=0.5` | ContextFlow schedule module | Verified/adapted | Upstream defaults to early 50 percent of steps. |
| Layer selection | `demo_insert.sh`, `demo_swap.sh` | ContextFlow explicit-layer schedule | Adapted | Do not transfer Wan layer IDs into LTX-2 defaults. |
| RF inversion | `wan/image2video.py:906-986`, `wan/utils/fm_solvers_modified.py:4` | ContextFlow inverter using LTX-2 second-order stepper | Adapted | Need LTX-2-compatible reverse schedule rather than copying Wan solver verbatim. |

## Expected Risks

- LTX-2 additive attention masks are already log-space bias tensors by the time attention runs; incorrect extension will silently corrupt attention.
- Source and target token layouts must match at every active step for K/V concatenation to be meaningful.
- `BatchSplitAdapter` may split large guidance batches internally, so branch metadata must be explicit and per-sample safe.
- Gradient checkpointing block calls will need the hook path to remain signature-compatible.

## Files Likely To Change

- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`
- `packages/ltx-pipelines/src/ltx_pipelines/contextflow/...`

