# Repository-first verification and implementation logging

## 1. Repository-first rule

Whenever any implementation detail is ambiguous, incomplete, or inconsistent, do not guess.

Inspect the actual repositories in this order:

1. the local LTX-2 codebase currently being modified;
2. the official ContextFlow repository:
   `https://github.com/yyChen233/ContextFlow`
3. the ContextFlow paper;
4. external documentation only when the first three sources are insufficient.

The local LTX-2 repository is authoritative for:

* class and function names;
* pipeline entry points;
* tensor shapes;
* token ordering;
* batch organization;
* scheduler behavior;
* timestep conventions;
* Q/K/V projection locations;
* normalization and RoPE order;
* attention backend;
* conditioning layout;
* device and dtype handling;
* configuration conventions;
* model block count;
* supported attention masks;
* existing extension or hook mechanisms.

The official ContextFlow repository is authoritative for:

* how ContextFlow structures inversion and generation;
* source-reconstruction and editing-path execution;
* source feature capture;
* Adaptive Context Enrichment;
* selected-layer activation;
* injection-step scheduling;
* source and target condition handling;
* ContextFlow-specific configuration and defaults.

The paper is authoritative for the intended algorithm and equations, but not necessarily for omitted engineering details.

Do not assume that a detail from the Wan-based ContextFlow implementation transfers directly to LTX-2.

---

## 2. Mandatory ambiguity protocol

When an ambiguity is encountered:

1. stop making assumptions about that detail;
2. search the local LTX-2 codebase;
3. inspect the corresponding implementation in the official ContextFlow repository;
4. compare both implementations;
5. record the finding;
6. choose an LTX-2-compatible design;
7. state whether the choice is:

   * directly verified;
   * adapted from ContextFlow;
   * inferred;
   * experimental.

Examples of ambiguities that require repository inspection include:

* whether K/V should be captured before or after Q/K normalization;
* whether K should be captured before or after RoPE;
* which tensor dimension is the token dimension;
* whether video and reference tokens share one sequence;
* whether source and target branches have identical token layouts;
* how classifier-free guidance or distilled guidance is batched;
* whether the scheduler expects velocity, noise, or another prediction type;
* whether inversion and forward sampling use reversed timestep conventions;
* how attention masks are represented;
* whether the attention backend accepts unequal query and key lengths;
* whether cached source K/V should include reference tokens;
* how block indices are assigned;
* whether IC-LoRA modifies the same hidden states or attention modules;
* whether source and target conditions can be passed through the same pipeline API.

Never resolve these only from memory or from a high-level paper diagram.

---

## 3. Required uncertainty reporting

When unsure, explicitly record the uncertainty rather than hiding it.

Use this format:

```text
Uncertainty:
The exact point at which LTX-2 applies video RoPE relative to K/V caching
has not yet been verified.

Required check:
Inspect <local file/class/function> and compare it with
<ContextFlow file/class/function>.

Implementation status:
Blocked until verified.
```

After verification, update it:

```text
Resolution:
LTX-2 applies Q/K normalization before 3D RoPE and calls the attention
backend immediately afterward.

Evidence:
- Local LTX-2: <file>:<symbol or line range>
- ContextFlow: <file>:<symbol or line range>

Decision:
Capture source K/V after normalization and RoPE, immediately before the
attention backend.

Confidence:
Directly verified.
```

If the repositories do not resolve the question, report:

```text
Resolution status:
Not specified by either repository.

Chosen experimental design:
...

Reason:
...

Alternative:
...

Risk:
...
```

Do not describe an inferred decision as paper-faithful or repository-verified.

---

## 4. Evidence requirements

For every important architectural or implementation claim, cite the exact code location.

Preferred format:

```text
Local LTX-2:
packages/.../attention.py
Class: VideoAttention
Method: forward

ContextFlow:
wan/modules/model.py
Class: WanSelfAttention
Method: forward
```

Include line numbers when stable, but always include file path and symbol because line numbers may change.

For every mapping from ContextFlow to LTX-2, maintain a table:

| ContextFlow operation | ContextFlow code location | LTX-2 integration point | Mapping confidence            | Notes |
| --------------------- | ------------------------- | ----------------------- | ----------------------------- | ----- |
| Source K/V capture    | `...`                     | `...`                   | Verified / adapted / inferred | `...` |
| K/V concatenation     | `...`                     | `...`                   | `...`                         | `...` |
| Active step window    | `...`                     | `...`                   | `...`                         | `...` |
| Layer selection       | `...`                     | `...`                   | `...`                         | `...` |
| RF inversion          | `...`                     | `...`                   | `...`                         | `...` |

Do not begin a major modification until its row has been completed.

---

## 5. Modification log

Maintain a persistent implementation log at:

```text
docs/contextflow/implementation_log.md
```

Append one entry for every meaningful code modification.

Use this format:

````markdown
## Change CF-001: Add identity attention-controller hook

Date:
Commit or working-tree state:

### Objective

Add a controller hook to LTX-2 video self-attention without changing
the model output.

### Files inspected

- `path/to/file.py`
  - `ClassName.method_name`
- `path/to/contextflow/file.py`
  - `ClassName.method_name`

### Files changed

- `path/to/file.py`
- `path/to/new_file.py`
- `tests/path/to/test_file.py`

### Code-block modifications

#### `path/to/file.py` — `ClassName.method_name`

Before:
- Q/K/V were passed directly to the attention backend.

After:
- Q/K/V are wrapped in `AttentionInputs`.
- The identity controller returns the tensors unchanged.

Reason:
- Establish a reusable intervention point before implementing ACE.

### Design choices

- Hook location: after Q/K normalization and RoPE.
- Scope: video self-attention only.
- Default behavior: controller disabled.
- Cache behavior: none in this change.

### Evidence

- Local LTX-2: `<file>:<symbol>`
- ContextFlow: `<file>:<symbol>`

### Ambiguities and resolutions

- Ambiguity:
- Repository checks:
- Resolution:
- Confidence:

### Tests run

```bash
<exact command>
````

Results:

* Passed:
* Failed:
* Skipped:

### No-op verification

* Reference output:
* Modified output:
* Maximum absolute error:
* Mean absolute error:
* Accepted tolerance:

### Runtime and memory impact

* Baseline peak VRAM:
* Modified peak VRAM:
* Baseline runtime:
* Modified runtime:

### Remaining risks

* ...

### Next recommended change

* ...

````

Do not overwrite earlier entries. The log should reflect the chronological implementation history.

---

## 6. File-change manifest

Maintain a concise current manifest at:

```text
docs/contextflow/file_manifest.md
````

Use this structure:

```markdown
# ContextFlow file manifest

## Added files

| File | Purpose | Status |
|---|---|---|
| `...` | `...` | Planned / implemented / tested |

## Modified files

| File | Symbols modified | Reason | Default behavior preserved |
|---|---|---|---|
| `...` | `...` | `...` | Yes / No |

## Tests

| File | Coverage |
|---|---|
| `...` | `...` |

## Documentation

| File | Purpose |
|---|---|
| `docs/contextflow/ltx2_attention_audit.md` | Architecture audit |
| `docs/contextflow/implementation_log.md` | Chronological changes |
| `docs/contextflow/design_decisions.md` | Accepted design decisions |
```

Update this manifest after every task.

---

## 7. Design-decision log

Maintain:

```text
docs/contextflow/design_decisions.md
```

Each non-trivial decision should receive an ID.

```markdown
## CF-D001: Capture source K/V after video RoPE

Status: Proposed / Accepted / Rejected / Superseded
Date:

### Context

ContextFlow requires source K/V from the reconstruction branch. LTX-2
normalizes and positionally encodes Q/K before attention.

### Evidence

- Local LTX-2: `<file>:<symbol>`
- ContextFlow: `<file>:<symbol>`
- Paper: `<section/equation>`

### Decision

Capture source K/V immediately before the attention backend, after Q/K
normalization and video RoPE.

### Rationale

This ensures source and target K use their correct positional encoding and
are represented at the same stage.

### Alternatives considered

1. Capture pre-RoPE K and reapply target RoPE.
2. Capture pre-normalization K.
3. Capture hidden states and recompute K/V.

### Risks

- Source and target token coordinate layouts must be identical.
- Additional conditioning tokens may need separate handling.

### Validation

- Shape assertions.
- No-op test.
- One-layer ACE test.

### Confidence

Directly verified / adapted / inferred / experimental.
```

Never silently change an accepted design. Add a new decision that supersedes the previous one.

---

## 8. Code-block modification reporting

At the end of every coding task, report modifications at the symbol or code-block level, not only by filename.

Required format:

```text
Modified:
- `path/to/file.py`
  - `ClassName.method_name`
    - Added branch metadata parameter.
    - Added controller call after RoPE.
    - Preserved original direct-attention path when controller is absent.

- `path/to/controller.py`
  - `ContextFlowAttentionController.record_source`
    - Added active-layer and active-step filtering.
  - `ContextFlowAttentionController.modify_target`
    - Added source K/V lookup and ACE invocation.

- `path/to/test_attention.py`
  - `test_identity_controller_matches_baseline`
  - `test_ace_augments_key_value_length`
```

For each modified block, state:

* previous behavior;
* new behavior;
* why it changed;
* whether default behavior is preserved;
* tests covering it.

Do not report only “implemented ContextFlow support.”

---

## 9. Required task-end report

Every local-agent task must end with:

```text
1. Repository state
   - LTX-2 commit:
   - ContextFlow commit:
   - Current branch:
   - Working-tree status:

2. Repository observations
   - Relevant files and symbols inspected:
   - Important verified tensor shapes:
   - Important control-flow findings:

3. Design decisions
   - New decisions:
   - Reused decisions:
   - Experimental assumptions:

4. Files changed
   - Added:
   - Modified:
   - Deleted:

5. Code-block modifications
   - File, symbol, previous behavior, new behavior, rationale.

6. Tests and commands
   - Exact commands:
   - Passed:
   - Failed:
   - Skipped:

7. No-op and regression status
   - Baseline comparison:
   - Numerical tolerance:
   - Any behavior changes with ContextFlow disabled:

8. Runtime and memory
   - Before:
   - After:
   - Measurement conditions:

9. Ambiguities
   - Resolved:
   - Unresolved:
   - Repository locations that still need inspection:

10. Risks
    - Correctness:
    - Memory:
    - Compatibility:
    - Research-faithfulness:

11. Documentation updated
    - Implementation log:
    - File manifest:
    - Design-decision log:
    - Architecture audit:

12. Next recommended task
```

A task is not complete until this report and the persistent logs are updated.

---

## 10. Restrictions against guessing

The local model must not:

* invent LTX-2 class or function names;
* assume tensor dimensions from the ContextFlow Wan implementation;
* assume the token dimension is `-2` without inspection;
* assume source and target sequence lengths are equal;
* assume post-RoPE capture is correct without checking LTX-2;
* assume attention masks can be extended by simple concatenation;
* assume ContextFlow’s layer indices transfer to LTX-2;
* assume LTX-2’s scheduler matches ContextFlow’s RF solver;
* assume the existing pipeline supports two branches without tracing it;
* implement from paper pseudocode when released code answers the question;
* claim a faithful reproduction when any central operation remains inferred.

When an assumption is unavoidable, make it configurable and log it as experimental.

---

## 11. Source priority for conflicting evidence

When sources disagree, follow this rule:

### LTX-2 engineering behavior

Use the local LTX-2 codebase.

Examples:

* tensor layout;
* scheduler API;
* attention backend;
* model structure;
* conditioning representation.

### ContextFlow algorithm behavior

Use the official ContextFlow repository, then the paper.

Examples:

* which branch supplies K/V;
* how ACE concatenates contexts;
* when injection is enabled;
* source-first target-second execution.

### LTX-2 adaptation choice

Document it as a design decision.

Do not alter LTX-2 merely to imitate Wan if the same algorithm can be expressed through the native LTX-2 architecture.

---

## 12. Initial repository-inspection checklist

Before making the first modification, inspect at minimum:

### Local LTX-2

* pipeline entry point used by the current RGB-video workflow;
* denoising loop;
* scheduler and timestep construction;
* transformer top-level forward;
* transformer block forward;
* video self-attention;
* Q/K/V projections;
* normalization;
* 3D RoPE;
* attention backend;
* IC-LoRA injection points;
* first-frame and source-video condition construction;
* model configuration containing block count and head dimensions.

### Official ContextFlow repository

* generation entry point;
* inversion path;
* synchronized or paired generation path;
* model-forward modifications;
* PnP/ACE layer activation;
* `pnp_layers` handling;
* `injection_step` handling;
* K/V concatenation;
* source and target prompt/first-frame handling;
* any repository defaults used in released examples.

Record all inspected paths and symbols in the architecture audit.

---

## 13. Guiding principle

The implementation process should follow:

```text
inspect
→ verify
→ document
→ decide
→ modify minimally
→ test
→ log
```

Never follow:

```text
infer from memory
→ make a broad patch
→ inspect only after failure
```
