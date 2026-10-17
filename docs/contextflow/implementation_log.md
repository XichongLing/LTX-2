# ContextFlow Implementation Log

## Change CF-001: Complete LTX-2 architecture audit before modification

Date: 2026-07-13
Commit or working-tree state: `6039e16435c330eac90ed64c60badb04990884cc` on branch `inversion`

### Objective

Trace the exact LTX-2 execution path needed for ContextFlow source K/V capture and target-side ACE injection before any code changes.

### Files inspected

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
  - `LTXModel._process_transformer_blocks`
  - `LTXModel.forward`
  - `X0Model.forward`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
  - `BasicAVTransformerBlock.__init__`
  - `BasicAVTransformerBlock.forward`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
  - `Attention.forward`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`
  - `PytorchPreAttention.__call__`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
  - `TransformerArgsPreprocessor._prepare_self_attention_mask`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py`
  - `_guided_denoise`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`
  - `modality_from_latent_state`
- `packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py`
  - `VideoConditionByLatentIndex.apply_to`
- `packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py`
  - `VideoConditionByKeyframeIndex.apply_to`
- `/tmp/ContextFlow/wan/modules/model.py`
  - `WanSelfAttention.forward`
- `/tmp/ContextFlow/wan/image2video.py`
  - `generate_with_pnp`
  - `generate_reconstruction`
- `/tmp/ContextFlow/wan/utils/fm_solvers_modified.py`
  - `FlowMatchNewScheduler`

### Files changed

- `docs/contextflow/ltx2_attention_audit.md`
- `docs/contextflow/design_decisions.md`
- `docs/contextflow/file_manifest.md`
- `docs/contextflow/implementation_log.md`

### Code-block modifications

#### `docs/contextflow/ltx2_attention_audit.md`

Before:
- File did not exist.

After:
- Added a repository-backed audit with exact integration points, uncertainties, and a ContextFlow-to-LTX-2 mapping table.

Reason:
- Required by the implementation guidelines before major code modification.

### Design choices

- Hook location: after Q/K normalization and RoPE, before the attention backend.
- Scope: video self-attention only.
- Source/target identity: explicit branch metadata, not implicit batched-pass ordering.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`
- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/ops.py:22`
- ContextFlow: `/tmp/ContextFlow/wan/modules/model.py:127`
- ContextFlow: `/tmp/ContextFlow/wan/image2video.py:455`

### Ambiguities and resolutions

- Ambiguity:
  - Whether LTX-2 captures the right representation before or after RoPE.
- Repository checks:
  - `PytorchPreAttention.__call__`
  - `Attention.forward`
- Resolution:
  - LTX-2 applies normalization first, then RoPE, then the backend call.
- Confidence:
  - Directly verified.

### Tests run

```bash
cd /home/xichong-drz-inokko/experimental/LTX-2 && git rev-parse HEAD
cd /tmp/ContextFlow && git rev-parse HEAD
```

Results:

- Passed: repository version capture
- Failed: none
- Skipped: code tests not started yet

## Change CF-002: Add identity-safe video self-attention hook plumbing

Date: 2026-07-13
Commit or working-tree state: `6039e16435c330eac90ed64c60badb04990884cc` on branch `inversion`

### Objective

Add a reusable controller seam to LTX-2 video self-attention so ContextFlow can record source K/V and modify target K/V without changing default behavior.

### Files inspected

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
  - `LTXModel._process_transformer_blocks`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
  - `BasicAVTransformerBlock.forward`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
  - `Attention.forward`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
  - `Modality`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
  - `TransformerArgs`

### Files changed

- `packages/ltx-core/src/ltx_core/guidance/attention_hooks.py`
- `packages/ltx-core/src/ltx_core/guidance/__init__.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`

### Code-block modifications

#### `packages/ltx-core/src/ltx_core/guidance/attention_hooks.py`

Before:
- No generic attention-hook protocol existed in `ltx-core`.

After:
- Added `SamplingStep`, `AttentionMetadata`, `AttentionInputs`, `AttentionController`, and `AttentionHookContext`.

Reason:
- `ltx-core` needs a dependency-safe interface that `ltx_pipelines.contextflow` can implement.

#### `packages/ltx-core/src/ltx_core/model/transformer/attention.py` — `Attention.forward`

Before:
- Q/K/V flowed directly from RoPE to the backend.

After:
- Optional source-record and target-modify controller calls run immediately before the backend.

Reason:
- Establish the minimal reusable intervention seam required by ContextFlow ACE.

### Design choices

- Hook location: post-RoPE, pre-backend.
- Scope: video self-attention only.
- Default behavior: unchanged when no hook context is provided.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`
- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/transformer.py:269`
- ContextFlow: `/tmp/ContextFlow/wan/modules/model.py:127`

### Ambiguities and resolutions

- Ambiguity:
  - Whether batch-guidance pass order could safely identify source vs target.
- Repository checks:
  - `packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py::_guided_denoise`
- Resolution:
  - Use explicit branch metadata instead of relying on batched CFG pass ordering.
- Confidence:
  - Directly verified.

### Tests run

```bash
/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python -m pytest /home/xichong-drz-inokko/experimental/LTX-2/tests/contextflow/test_attention_noop.py -q
```

Results:

- Passed: no-op source and target hook equivalence
- Failed: none
- Skipped: broader contextflow package tests not yet added

## Change CF-003: Add ContextFlow package, sampler, inverter, and tests

Date: 2026-07-13
Commit or working-tree state: `6039e16435c330eac90ed64c60badb04990884cc` on branch `inversion`

### Objective

Implement the first modular ContextFlow package on top of the new attention seam, including schedules, feature storage, ACE policy, inversion, dual-trajectory sampling, and smoke-test coverage.

### Files inspected

- `packages/ltx-core/src/ltx_core/tools.py`
  - `VideoLatentTools.create_initial_state`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`
  - `modality_from_latent_state`
- `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`
  - `DiffusionStage.model_context`
  - `VideoDecoder.__call__`

### Files changed

- `packages/ltx-pipelines/src/ltx_pipelines/contextflow/...`
- `packages/ltx-pipelines/src/ltx_pipelines/__init__.py`
- `tests/contextflow/...`

### Code-block modifications

#### `packages/ltx-pipelines/src/ltx_pipelines/contextflow/trajectory.py` — `DualTrajectorySampler.sample`

Before:
- No synchronized source/target trajectory runner existed.

After:
- Added a sequential source-then-target midpoint sampler that shares the inverted noise anchor and clears the feature cache every step.

Reason:
- Match the required dual-trajectory execution order for ContextFlow.

#### `packages/ltx-pipelines/src/ltx_pipelines/contextflow/inversion.py` — `RectifiedFlowInverter.invert`

Before:
- No ContextFlow-specific inversion module existed.

After:
- Added a reverse-time midpoint inverter that excludes terminal sigma `0.0` from the reverse loop.

Reason:
- Provide an LTX-2-compatible high-order inversion scaffold without copying Wan code verbatim.

### Design choices

- Stepper: midpoint RF-style ODE update.
- Reverse schedule: reversed non-terminal sigmas only.
- Pipeline guidance: minimal prompt-context path first; CFG-specific expansion remains future work.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/components/schedulers.py:18`
- Local LTX-2: `packages/ltx-core/src/ltx_core/tools.py:89`
- ContextFlow: `/tmp/ContextFlow/wan/image2video.py:906`
- ContextFlow: `/tmp/ContextFlow/wan/utils/fm_solvers_modified.py:4`

### Ambiguities and resolutions

- Ambiguity:
  - Whether to start inversion from exact sigma `0.0`.
- Repository checks:
  - LTX-2 scheduler terminal value
  - ContextFlow solver minimum sigma handling
- Resolution:
  - Exclude terminal zero from the reverse loop.
- Confidence:
  - Adapted.

### Tests run

```bash
/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python -m pytest /home/xichong-drz-inokko/experimental/LTX-2/tests/contextflow -q
```

Results:

- Passed: 15 tests
- Failed: none
- Skipped: heavy model-backed runtime experiments

## Change CF-004: Stabilize large-token sigmas and pair ContextFlow endpoints

Date: 2026-07-22
Commit or working-tree state: dirty working tree on branch `inversion`

### Objective

Prevent large video latents from producing duplicate or NaN sigma schedules and use the same positive sigma grid for ContextFlow inversion and reconstruction.

### Files inspected

- `packages/ltx-core/src/ltx_core/components/schedulers.py`
  - `LTX2Scheduler.execute`
- `packages/ltx-pipelines/src/ltx_pipelines/inversion/inversion.py`
  - `RectifiedFlowInverter.invert`
- `/home/xichong-drz-inokko/experimental/ContextFlow/wan/utils/fm_solvers_modified.py`
  - `FlowMatchNewScheduler`
- `/home/xichong-drz-inokko/experimental/ContextFlow/wan/image2video.py`
  - inversion and reconstruction loops

### Files changed

- `packages/ltx-core/src/ltx_core/components/schedulers.py`
- `packages/ltx-pipelines/src/ltx_pipelines/inversion/schedule.py`
- `packages/ltx-pipelines/src/ltx_pipelines/inversion/inversion.py`
- `packages/ltx-pipelines/src/ltx_pipelines/inversion/reconstruction.py`
- `packages/ltx-pipelines/src/ltx_pipelines/inversion/pipeline.py`
- `scripts/inversion/debug_contextflow_constant_flow.py`
- `scripts/inversion/debug_contextflow_linear_flow_convergence.py`
- `scripts/inversion/debug_contextflow_linear_flow_full_sample.py`
- `tests/contextflow/test_schedule.py`

### Code-block modifications

#### `LTX2Scheduler.execute`

- Builds shifted and stretched sigmas in float64 using a stable logit-space shift.
- Converts to float32 only after schedule construction.
- Raises if stretching collapses the usable sigma range.

#### `contextflow_sigmas` and `RectifiedFlowInverter.invert`

- Builds `steps + 1` positive sigmas ending at ContextFlow's post-shift minimum, approximately `0.01479`, for exactly `steps` transitions.
- Validates that the grid is finite, positive, and strictly decreasing.
- Reverses the complete positive grid for inversion and removes the asymmetric synthetic `0 -> 0.1` lift.

### Design choices

- The nonzero endpoint applies ContextFlow's shift `5.0` to its raw minimum `0.003 / 1.002`.
- Token-dependent shifting remains LTX-2-specific rather than copying Wan's fixed shift.
- Invalid zero-ended schedules now fail explicitly in the ContextFlow inverter.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/components/schedulers.py:LTX2Scheduler.execute`
- ContextFlow: `wan/utils/fm_solvers_modified.py:FlowMatchNewScheduler.set_timesteps`
- ContextFlow: `wan/image2video.py` inversion and reconstruction loops

### Tests run

```bash
python -m pytest -q tests/contextflow/test_schedule.py
python scripts/inversion/debug_contextflow_constant_flow.py --steps 10 --dtype float32 --step-compute-dtype float32
python scripts/inversion/debug_contextflow_linear_flow_convergence.py --steps 10,20,40,80 --dtype float64
python -m py_compile <changed Python files>
git diff --check
```

Results:

- Schedule tests: 4 passed.
- Building-sized latent schedule: finite and strictly decreasing.
- Constant-flow maximum roundtrip error: `1.192092896e-07`.
- First-order linear-flow slope: approximately `-0.99`.
- Syntax and whitespace checks passed.
