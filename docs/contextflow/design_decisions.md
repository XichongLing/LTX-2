# ContextFlow Design Decisions

## CF-D001: Capture source K/V after video RoPE

Status: Accepted
Date: 2026-07-13

### Context

ContextFlow requires source K/V from the reconstruction branch. LTX-2 normalizes and positionally encodes Q/K before attention.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/ops.py:22`
- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/attention.py:529`
- ContextFlow: `wan/modules/model.py:127`
- Paper: Section 3.4, Figure 3

### Decision

Capture source K/V immediately before the attention backend, after Q/K normalization and video RoPE.

### Rationale

This keeps source and target keys at the same representation stage and avoids reapplying RoPE externally.

### Alternatives Considered

1. Capture pre-RoPE K and reapply target RoPE later.
2. Capture pre-normalization K.
3. Capture hidden states and recompute K/V outside attention.

### Risks

- Source and target token coordinate layouts must be identical.
- Attention-mask extension must match LTX-2's additive-bias representation.

### Validation

- No-op controller tests.
- One-layer ACE shape tests.
- Dual-trajectory cache lifecycle tests.

### Confidence

Directly verified.

## CF-D003: Exclude terminal sigma=0 from the reverse inversion loop

Status: Accepted
Date: 2026-07-13

### Context

The local LTX-2 schedule ends at an exact terminal sigma of `0.0`. Reversing that schedule naively would start inversion at `sigma=0`, which makes the velocity conversion singular.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/components/schedulers.py:18`
- Local implementation: `packages/ltx-pipelines/src/ltx_pipelines/contextflow/inversion.py:27`
- ContextFlow: `/tmp/ContextFlow/wan/utils/fm_solvers_modified.py:4`

### Decision

Use the reversed schedule without the terminal zero element for inversion.

### Rationale

This preserves the reverse-time behavior while avoiding a divide-by-zero at the start of inversion.

### Alternatives Considered

1. Clamp zero to a fixed epsilon.
2. Start from the full reversed schedule and special-case the first step.

### Risks

- The reverse schedule is adapted to LTX-2 and is not a byte-for-byte reproduction of the Wan solver.

### Validation

- `tests/contextflow/test_pipeline_smoke.py`
- `tests/contextflow/test_dual_trajectory.py`

### Confidence

Adapted.

## CF-D002: Extend the additive self-attention bias inside the hook path

Status: Accepted
Date: 2026-07-13

### Context

LTX-2 converts raw self-attention masks in `[0, 1]` into additive log-space bias before transformer blocks run.

### Evidence

- Local LTX-2: `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py:147`
- Local LTX-2: `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:262`
- Paper: Section 3.4

### Decision

ContextFlow will extend the prepared additive bias tensor at the attention-hook boundary rather than trying to reconstruct the raw mask format.

### Rationale

The attention backend consumes the additive-bias format directly. Extending masks earlier would couple the implementation to more of the preprocessing stack.

### Alternatives Considered

1. Extend only raw `(B, T, T)` masks before preprocessing.
2. Ignore masks during ACE and rely on unrestricted attention.

### Risks

- Mixed boolean/additive inputs still need tests at the ACE policy layer.

### Validation

- Mask extension unit tests for `None`, boolean, and additive masks.

### Confidence

Directly verified.
