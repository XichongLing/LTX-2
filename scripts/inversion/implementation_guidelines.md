# ContextFlow-First Implementation Framework for LTX-2

## 1. Objective

Implement ContextFlow on the current LTX-2 codebase using a minimal modular framework.

The implementation must support:

* source-video rectified-flow inversion;
* synchronized source-reconstruction and target-editing trajectories;
* source K/V capture from selected LTX-2 video self-attention blocks;
* ContextFlow Adaptive Context Enrichment through K/V concatenation;
* configurable active layers and denoising steps;
* no-op equivalence with the original LTX-2 pipeline;
* logging and ablations.

Although a second algorithm may be added later, this task must contain only ContextFlow-specific behavior.

---

# 2. Minimal architecture

Use the following decomposition:

```text
ContextFlowPipeline
├── Inverter
├── DualTrajectorySampler
├── AttentionController
│   ├── FeatureStore
│   └── ContextFlowACEPolicy
├── LayerStepSchedule
└── Instrumentation
```

Do not create a large generic research framework yet.

The important separation is:

```text
pipeline orchestration
    ↓
dual trajectory execution
    ↓
source feature recording
    ↓
target attention modification
```

---

# 3. Proposed file layout

Create a focused package:

```text
packages/ltx-pipelines/src/ltx_pipelines/contextflow/
├── __init__.py
├── config.py
├── pipeline.py
├── inversion.py
├── trajectory.py
├── schedule.py
├── cli.py
│
├── attention/
│   ├── __init__.py
│   ├── controller.py
│   ├── feature_store.py
│   └── ace.py
│
└── instrumentation/
    ├── __init__.py
    ├── tensor_stats.py
    └── memory.py

tests/contextflow/
├── test_feature_store.py
├── test_schedule.py
├── test_ace_attention.py
├── test_attention_noop.py
├── test_dual_trajectory.py
└── test_pipeline_smoke.py
```

---

# 4. Generic interfaces to preserve

Only preserve the following reusable interfaces.

## 4.1 Inverter

```python
class Inverter(Protocol):
    def invert(
        self,
        *,
        source_latent: torch.Tensor,
        source_condition: "BranchCondition",
    ) -> "InversionResult":
        ...
```

```python
@dataclass
class InversionResult:
    initial_noise: torch.Tensor
    timesteps: torch.Tensor
    trajectory: list[torch.Tensor] | None = None
```

For ContextFlow, the implementation should use a high-order RF inversion method.

The optional `trajectory` field should exist but remain unused by ContextFlow.

Do not implement trajectory-dependent corrections.

---

## 4.2 Attention controller

```python
class AttentionController(Protocol):
    def begin_step(self, step: "SamplingStep") -> None:
        ...

    def record_source(
        self,
        *,
        step: "SamplingStep",
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: "AttentionMetadata",
    ) -> None:
        ...

    def modify_target(
        self,
        *,
        step: "SamplingStep",
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        metadata: "AttentionMetadata",
    ) -> "AttentionInputs":
        ...

    def end_step(self, step: "SamplingStep") -> None:
        ...
```

This is the main interface that should remain reusable later.

Do not include source Q capture in the initial implementation.

---

## 4.3 Attention inputs

```python
@dataclass
class AttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_mask: torch.Tensor | None
```

The model attention code should consume this return type without knowing which intervention policy produced it.

---

## 4.4 Step and layer schedule

```python
class InterventionSchedule(Protocol):
    def is_active(
        self,
        *,
        step_index: int,
        num_steps: int,
        layer_index: int,
    ) -> bool:
        ...
```

This abstraction will later allow other methods to use different layer and timestep schedules.

For now, implement only explicit layer sets and a normalized denoising interval.

---

# 5. ContextFlow-specific modules

## 5.1 ContextFlow pipeline

```python
class ContextFlowPipeline:
    def __init__(
        self,
        *,
        base_pipeline,
        inverter: Inverter,
        trajectory_sampler: "DualTrajectorySampler",
        attention_controller: "ContextFlowAttentionController",
        instrumentation: "ContextFlowInstrumentation",
    ):
        ...
```

Expected call:

```python
result = pipeline(
    source_video=source_video,
    source_first_frame=source_first_frame,
    edited_first_frame=edited_first_frame,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
    seed=seed,
)
```

Return:

```python
@dataclass
class ContextFlowResult:
    edited_video: torch.Tensor
    reconstructed_video: torch.Tensor | None
    metrics: dict[str, float]
    debug_artifacts: dict[str, str]
```

---

## 5.2 Dual trajectory sampler

The sampler should be generic enough to run two synchronized trajectories but must not contain algorithm-specific attention logic.

```python
class DualTrajectorySampler:
    def sample(
        self,
        *,
        initial_noise: torch.Tensor,
        source_condition: "BranchCondition",
        target_condition: "BranchCondition",
        timesteps: torch.Tensor,
        attention_controller: AttentionController,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_latent = initial_noise.clone()
        target_latent = initial_noise.clone()

        for step_index, timestep in enumerate(timesteps):
            step = SamplingStep(
                index=step_index,
                timestep=timestep,
                num_steps=len(timesteps),
            )

            attention_controller.begin_step(step)

            source_prediction = self.forward_source(
                latent=source_latent,
                timestep=timestep,
                condition=source_condition,
                attention_controller=attention_controller,
            )

            target_prediction = self.forward_target(
                latent=target_latent,
                timestep=timestep,
                condition=target_condition,
                attention_controller=attention_controller,
            )

            source_latent = self.scheduler.step(
                source_prediction,
                timestep,
                source_latent,
            )

            target_latent = self.scheduler.step(
                target_prediction,
                timestep,
                target_latent,
            )

            attention_controller.end_step(step)

        return source_latent, target_latent
```

Important invariants:

* source forward always runs before target forward;
* both branches use the same timestep;
* both start from cloned copies of the same inverted latent;
* feature cache is cleared after each target forward;
* the source and target scheduler updates are independent;
* no source latent is copied into the target latent after initialization.

---

# 6. Attention integration point

Before modifying code, inspect and document the exact LTX-2 video self-attention path.

Identify:

* transformer block class;
* video self-attention module;
* Q/K/V projection lines;
* Q/K normalization;
* video RoPE application;
* SDPA or FlashAttention call;
* sequence dimension;
* head dimension;
* batch organization;
* guidance branch organization;
* whether conditioning/reference tokens participate in video self-attention.

LTX-2 currently exposes separate transformer attention operations and supports attention perturbations, so the implementation should extend the existing attention path rather than duplicate the full transformer.

The first implementation should capture and concatenate K/V at the point immediately before the attention backend call, after any required normalization and positional encoding.

Target structure:

```python
q, k, v = compute_qkv(hidden_states)
q, k = apply_qk_norm(q, k)
q, k = apply_video_rope(q, k)

if branch == "source":
    controller.record_source(
        step=step,
        layer_index=layer_index,
        k=k,
        v=v,
        metadata=metadata,
    )

if branch == "target":
    attention_inputs = controller.modify_target(
        step=step,
        layer_index=layer_index,
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
        metadata=metadata,
    )
else:
    attention_inputs = AttentionInputs(
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
    )

output = attention_backend(
    attention_inputs.q,
    attention_inputs.k,
    attention_inputs.v,
    attention_inputs.attention_mask,
)
```

Do not modify:

* text cross-attention;
* audio self-attention;
* audio-video cross-attention;
* VAE;
* IC-LoRA condition construction.

Only modify video self-attention.

---

# 7. Feature store

```python
@dataclass
class StoredKV:
    k: torch.Tensor
    v: torch.Tensor
    step_index: int
    layer_index: int
    representation_stage: str
```

```python
class FeatureStore:
    def __init__(self):
        self._cache: dict[tuple[int, int], StoredKV] = {}

    def put(
        self,
        *,
        step_index: int,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
        representation_stage: str,
    ) -> None:
        ...

    def get(
        self,
        *,
        step_index: int,
        layer_index: int,
    ) -> StoredKV:
        ...

    def clear_step(self, step_index: int) -> None:
        ...

    def clear_all(self) -> None:
        ...
```

Initial storage policy:

* store only K and V;
* store only active layers;
* store only active denoising steps;
* preserve original dtype;
* preserve original device;
* detach only if gradients are definitely not required;
* do not move tensors to CPU during the first version;
* clear after target forward.

Assertions:

```python
assert k.shape[-2] == v.shape[-2]
assert k.shape[-1] == v.shape[-1]
assert torch.isfinite(k).all()
assert torch.isfinite(v).all()
```

Do not store attention probabilities.

---

# 8. ContextFlow ACE policy

ContextFlow retains target Q and concatenates source K/V with target K/V along the token dimension. The paper presents this as soft contextual enrichment rather than hard source-feature replacement.

```python
class ContextFlowACEPolicy:
    def __init__(
        self,
        *,
        schedule: InterventionSchedule,
        concat_order: str = "target_source",
    ):
        self.schedule = schedule
        self.concat_order = concat_order

    def apply(
        self,
        *,
        step: SamplingStep,
        layer_index: int,
        target_q: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> AttentionInputs:
        if not self.schedule.is_active(
            step_index=step.index,
            num_steps=step.num_steps,
            layer_index=layer_index,
        ):
            return AttentionInputs(
                q=target_q,
                k=target_k,
                v=target_v,
                attention_mask=attention_mask,
            )

        self.validate_shapes(
            target_q=target_q,
            target_k=target_k,
            target_v=target_v,
            source_k=source_k,
            source_v=source_v,
        )

        if self.concat_order == "target_source":
            k_aug = torch.cat(
                [target_k, source_k],
                dim=TOKEN_DIM,
            )
            v_aug = torch.cat(
                [target_v, source_v],
                dim=TOKEN_DIM,
            )
        else:
            k_aug = torch.cat(
                [source_k, target_k],
                dim=TOKEN_DIM,
            )
            v_aug = torch.cat(
                [source_v, target_v],
                dim=TOKEN_DIM,
            )

        mask_aug = extend_attention_mask(
            attention_mask,
            added_key_tokens=source_k.shape[TOKEN_DIM],
        )

        return AttentionInputs(
            q=target_q,
            k=k_aug,
            v=v_aug,
            attention_mask=mask_aug,
        )
```

Required shape checks:

```python
assert target_k.shape[:-2] == source_k.shape[:-2]
assert target_v.shape[:-2] == source_v.shape[:-2]

assert target_k.shape[-1] == source_k.shape[-1]
assert target_v.shape[-1] == source_v.shape[-1]

assert target_k.shape[-2] == target_v.shape[-2]
assert source_k.shape[-2] == source_v.shape[-2]
```

The target query length must remain unchanged.

---

# 9. ContextFlow attention controller

```python
class ContextFlowAttentionController:
    def __init__(
        self,
        *,
        feature_store: FeatureStore,
        ace_policy: ContextFlowACEPolicy,
        active_layers: set[int],
    ):
        self.feature_store = feature_store
        self.ace_policy = ace_policy
        self.active_layers = active_layers

    def record_source(
        self,
        *,
        step,
        layer_index,
        k,
        v,
        metadata,
    ):
        if layer_index not in self.active_layers:
            return

        if not self.ace_policy.schedule.is_active(
            step_index=step.index,
            num_steps=step.num_steps,
            layer_index=layer_index,
        ):
            return

        self.feature_store.put(
            step_index=step.index,
            layer_index=layer_index,
            k=k,
            v=v,
            representation_stage=metadata.representation_stage,
        )

    def modify_target(
        self,
        *,
        step,
        layer_index,
        q,
        k,
        v,
        attention_mask,
        metadata,
    ):
        if layer_index not in self.active_layers:
            return AttentionInputs(q, k, v, attention_mask)

        if not self.ace_policy.schedule.is_active(
            step_index=step.index,
            num_steps=step.num_steps,
            layer_index=layer_index,
        ):
            return AttentionInputs(q, k, v, attention_mask)

        stored = self.feature_store.get(
            step_index=step.index,
            layer_index=layer_index,
        )

        if stored.representation_stage != metadata.representation_stage:
            raise RuntimeError(
                "Source and target K/V were captured at "
                "different representation stages."
            )

        return self.ace_policy.apply(
            step=step,
            layer_index=layer_index,
            target_q=q,
            target_k=k,
            target_v=v,
            source_k=stored.k,
            source_v=stored.v,
            attention_mask=attention_mask,
        )

    def end_step(self, step):
        self.feature_store.clear_step(step.index)
```

---

# 10. Configuration

Use a compact ContextFlow-only configuration.

```yaml
method: contextflow

input:
  source_video: null
  source_first_frame: null
  edited_first_frame: null

prompts:
  source: ""
  target: ""

inversion:
  type: rf_solver
  order: 2
  num_steps: 50

dual_trajectory:
  enabled: true
  execution: sequential
  share_initial_noise: true

ace:
  enabled: true

  layers:
    type: explicit
    indices: [4]

  steps:
    start_fraction: 0.0
    end_fraction: 0.5

  concat_order: target_source
  representation_stage: post_rope

debug:
  tensor_statistics: true
  measure_memory: true
  save_reconstruction: true
  use_debug_attention: false
```

Initial default should use only one active layer.

Do not begin with a copied ContextFlow Wan layer set.

---

# 11. Required implementation phases

## Phase 1: repository audit

No code changes.

Produce:

```text
docs/contextflow/ltx2_attention_audit.md
```

It must contain:

* exact LTX-2 commit hash;
* pipeline entry point;
* denoising-loop function;
* transformer forward path;
* video self-attention class;
* Q/K/V shape;
* Q/K normalization location;
* RoPE location;
* attention backend;
* sequence dimension;
* block count;
* guidance batch layout;
* source-video conditioning path;
* edited-first-frame conditioning path;
* proposed hook location.

Do not proceed until this audit is complete.

---

## Phase 2: identity controller

Add the controller hook with no feature recording or modification.

Acceptance tests:

```text
controller disabled == original implementation
identity controller == original implementation
empty active layers == original implementation
```

Use deterministic inputs and compare tensors or decoded outputs with an appropriate tolerance.

---

## Phase 3: feature recorder

Record source K/V from:

```text
one block
one denoising step
```

Deliver logging for:

* shape;
* dtype;
* device;
* norm;
* memory size;
* representation stage.

No target modification yet.

---

## Phase 4: dual trajectory scaffold

Implement source and target trajectories with no ACE.

Tests:

* both begin from identical inverted noise;
* source and target branches use identical timesteps;
* source and target latents become independent after initialization;
* feature cache is cleared;
* source reconstruction can be decoded;
* target result matches a dual-path run with identity controller.

---

## Phase 5: one-layer ACE

Enable K/V concatenation at:

```text
one block
one denoising step
```

Verify:

* target Q length unchanged;
* target K/V length increases by source sequence length;
* output sequence length unchanged;
* no NaNs;
* attention mask is correctly extended;
* inactive blocks remain identical to the dual-path baseline.

---

## Phase 6: step and layer schedules

Support:

* explicit layer list;
* layer range;
* normalized start/end step fractions.

Do not implement automatic vital-layer analysis yet.

---

## Phase 7: full ContextFlow run

Run a short, low-resolution experiment with:

```text
1 active layer
first 25% of steps
```

Then expand to:

```text
4 active layers
first 50% of steps
```

Measure:

* source reconstruction quality;
* edited first-frame adherence;
* temporal consistency;
* runtime;
* peak VRAM.

---

## Phase 8: layer-response tooling

After ACE is stable, implement layer ablations.

For each candidate layer:

1. run the dual-path baseline;
2. run ACE only at that layer;
3. calculate output change;
4. calculate reconstruction retention;
5. calculate edit adherence;
6. record runtime and memory.

Do not assume Wan’s vital layers transfer to LTX-2.

---

# 12. Essential ablations

Support these through configuration, without separate code paths.

```text
baseline LTX-2
dual trajectory without ACE
ACE at one layer
ACE at multiple layers
ACE during early 25%
ACE during early 50%
ACE during all steps
target K/V only
target + source K only
target + source V only
target + source K/V
source K/V replacement
source K/V concatenation
```

Only the final two are needed to compare hard replacement against ContextFlow’s enrichment mechanism.

---

# 13. Instrumentation

For every active layer and step record:

```text
step index
normalized step progress
layer index
target Q shape
target K/V shape
source K/V shape
augmented K/V shape
dtype
device
source cache size
attention output norm
source-target K cosine similarity
source-target V cosine similarity
```

Optionally, for debug-only low-resolution runs, record the fraction of attention mass assigned to source tokens.

Do not enable explicit attention-weight extraction during normal inference.

The optimized attention implementation should remain the default. LTX-2’s attention backend has active optimization work, so replacing it globally with manual softmax attention could distort runtime and memory measurements.

---

# 14. Tests

## Unit tests

### Feature store

```text
put/get works
wrong step raises
wrong layer raises
clear_step removes only the current step
clear_all removes all entries
```

### ACE shape test

Use synthetic tensors:

```python
B, H, Nt, Ns, D = 1, 4, 8, 8, 16
```

Verify:

```text
Q output length = Nt
K/V input length = Nt + Ns
attention output length = Nt
```

### Schedule test

Verify exact boundary behavior at:

```text
0%
25%
50%
100%
```

### Mask extension

Test:

* no mask;
* boolean mask;
* additive mask;
* causal or structured mask, when applicable.

---

## Integration tests

### No-op equivalence

```text
ACE disabled
empty active layers
zero-length active interval
```

All should reproduce the dual-path baseline.

### One-layer intervention

Confirm that only the requested layer receives augmented K/V.

### Cache lifecycle

After each denoising step:

```python
assert feature_store.is_empty()
```

### Repeated invocation

Run the pipeline twice and confirm no stale source features remain.

---

# 15. Memory constraints

The first implementation should prioritize correctness.

Use:

```yaml
memory:
  source_target_execution: sequential
  cache_device: cuda
  cache_dtype: model_dtype
  clear_after_target_forward: true
  cache_only_active_layers: true
  cache_only_active_steps: true
```

Do not initially:

* batch source and target together;
* offload K/V to CPU;
* quantize cached K/V;
* retain all-step features;
* extract attention probabilities.

Later optimizations can remain compatible with the `FeatureStore` interface.

---

# 16. Explicit non-goals

The local agent must not:

* implement FreeViS;
* add multiple references;
* add optical flow;
* add source Q capture;
* add FFT filters;
* add AdaIN;
* add latent correction;
* add style-specific abstractions;
* redesign LTX conditioning;
* refactor unrelated pipelines;
* change model weights;
* train a LoRA;
* modify audio attention;
* copy Wan-specific block indices into LTX-2;
* claim ContextFlow reproduction before testing inversion and ACE separately.

---

# 17. Preserved future extension points

Although only ContextFlow is implemented, preserve these interfaces:

```text
Inverter
AttentionController
AttentionInputs
InterventionSchedule
FeatureStore
DualTrajectorySampler
```

Later work may add:

* another attention controller;
* another latent post-step controller;
* another reference manager;
* trajectory-dependent correction.

Do not implement those modules now.

The present code should not contain comments such as:

```text
TODO FreeViS
```

Future compatibility should come from clean interfaces, not unused placeholder classes.

---

# 18. Recommended first task for the local agent

## Task title

Audit LTX-2 for ContextFlow ACE integration

## Objective

Trace the exact LTX-2 execution path required to implement a synchronized reconstruction/editing trajectory and source K/V enrichment in video self-attention.

## Required output

Create:

```text
docs/contextflow/ltx2_attention_audit.md
```

Include:

1. exact repository commit;
2. pipeline entry point currently used;
3. inversion and denoising scheduler path;
4. transformer invocation;
5. video self-attention implementation;
6. Q/K/V shapes;
7. normalization and RoPE order;
8. attention backend;
9. block indexing;
10. token and batch layout;
11. candidate hook point;
12. expected risks;
13. files likely to change.

## Restrictions

* Make no code changes.
* Do not design FreeViS.
* Do not assume Wan and LTX-2 share layer behavior.
* Do not infer tensor shapes without tracing or logging.
* Mark every unverified assumption explicitly.

## Completion criteria

The audit is complete only when another coding agent can identify exactly where to:

* record source K/V;
* retrieve source K/V during target forward;
* concatenate K/V before attention;
* extend the attention mask;
* clear the cache after the target forward.

````

---

# 19. Final implementation order

Use this exact sequence:

```text
1. LTX-2 architecture audit
2. identity attention controller
3. source K/V recorder
4. synchronized dual trajectory
5. one-step one-layer ACE
6. configurable ACE schedule
7. RF-Solver inversion verification
8. complete ContextFlow inference
9. layer-response analysis
10. memory optimization
````

Do not combine steps 2–7 in one agent task.

The ContextFlow implementation is complete when:

* the original pipeline is unchanged by default;
* source and target paths begin from the same inverted latent;
* source K/V are captured from the correct video self-attention representation;
* target Q attends to concatenated target/source K/V;
* intervention is restricted by layer and timestep schedules;
* source caches are cleared safely;
* reconstruction and edited outputs are saved;
* no-op and shape tests pass;
* peak memory and runtime are reported.
