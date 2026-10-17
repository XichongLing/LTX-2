# ContextFlow file manifest

## Added files

| File | Purpose | Status |
|---|---|---|
| `docs/contextflow/ltx2_attention_audit.md` | Repository-backed LTX-2 integration audit | implemented |
| `docs/contextflow/implementation_log.md` | Chronological implementation log | implemented |
| `docs/contextflow/design_decisions.md` | Accepted design decisions | implemented |
| `packages/ltx-core/src/ltx_core/guidance/attention_hooks.py` | Generic attention-hook protocol and metadata | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/__init__.py` | ContextFlow package exports | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/config.py` | ContextFlow configuration and branch-condition types | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/schedule.py` | Layer and step activation schedule | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/attention/__init__.py` | Attention subpackage exports | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/attention/feature_store.py` | Source K/V cache | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/attention/ace.py` | ACE policy and attention-mask extension | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/attention/controller.py` | No-op and ContextFlow attention controllers | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/instrumentation/__init__.py` | Attention instrumentation collector | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/instrumentation/tensor_stats.py` | Tensor statistics helpers | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/instrumentation/memory.py` | Tensor memory helpers | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/trajectory.py` | Dual-trajectory midpoint sampler | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/inversion.py` | Rectified-flow-style inversion scaffold | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/pipeline.py` | End-to-end ContextFlow pipeline | implemented |
| `packages/ltx-pipelines/src/ltx_pipelines/contextflow/cli.py` | Minimal CLI entrypoint | implemented |
| `tests/contextflow/test_feature_store.py` | Feature store unit tests | implemented |
| `tests/contextflow/test_schedule.py` | Schedule unit tests | implemented |
| `tests/contextflow/test_ace_attention.py` | ACE policy and mask tests | implemented |
| `tests/contextflow/test_attention_noop.py` | Attention no-op equivalence tests | implemented |
| `tests/contextflow/test_dual_trajectory.py` | Dual-trajectory sampler test | implemented |
| `tests/contextflow/test_pipeline_smoke.py` | Pipeline smoke test | implemented |

## Modified files

| File | Symbols modified | Reason | Default behavior preserved |
|---|---|---|---|
| `packages/ltx-core/src/ltx_core/guidance/__init__.py` | module exports | Export generic attention-hook interfaces | Yes |
| `packages/ltx-core/src/ltx_core/model/transformer/modality.py` | `Modality`, `Modality.split` | Carry optional attention-hook context through model inputs | Yes |
| `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py` | `TransformerArgs`, `TransformerArgsPreprocessor.prepare` | Preserve hook metadata into transformer blocks | Yes |
| `packages/ltx-core/src/ltx_core/model/transformer/attention.py` | `Attention.forward` | Add opt-in source record / target modify seam before backend call | Yes |
| `packages/ltx-core/src/ltx_core/model/transformer/transformer.py` | `BasicAVTransformerBlock.forward` | Thread block index and hook context into video self-attention | Yes |
| `packages/ltx-core/src/ltx_core/model/transformer/model.py` | `LTXModel._process_transformer_blocks` | Pass block index through normal and checkpointed execution | Yes |
| `packages/ltx-pipelines/src/ltx_pipelines/__init__.py` | module exports | Export `ContextFlowPipeline` | Yes |

## Tests

| File | Coverage |
|---|---|
| `tests/contextflow/test_feature_store.py` | feature-store cache lifecycle |
| `tests/contextflow/test_schedule.py` | exact step/layer boundary activation |
| `tests/contextflow/test_ace_attention.py` | K/V concatenation and mask extension |
| `tests/contextflow/test_attention_noop.py` | no-op attention-hook equivalence |
| `tests/contextflow/test_dual_trajectory.py` | source/target divergence and cache clearing |
| `tests/contextflow/test_pipeline_smoke.py` | end-to-end package smoke path |

## Documentation

| File | Purpose |
|---|---|
| `docs/contextflow/ltx2_attention_audit.md` | Architecture audit |
| `docs/contextflow/implementation_log.md` | Chronological changes |
| `docs/contextflow/design_decisions.md` | Accepted design decisions |
