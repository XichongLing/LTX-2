# Reference-injection segmentation check

Checks whether LTX-2.3 + edit-anything IC-LoRA produces semantically-grounded
cross-attention: each text token should attend to the correct spatial region.

## Files

| File | Purpose |
|---|---|
| `attn_capture.py` | Captures cross-attention weights via SDPA monkeypatch |
| `mask_from_attn.py` | Offline: aggregation → spatial mask (no LTX import, unit-testable) |
| `overlay.py` | Offline: heatmap/mask compositing onto video frames |
| `prompt_token_utils.py` | Token inspection and word-to-index lookup for Gemma |
| `run_reference_mask_check.py` | Runner: discover / sweep / mask modes |
| `launch.sh` | Convenience launcher — edit paths, then `bash launch.sh <mode>` |

## Token–pixel mapping (key fact)

The transformer operates on flat latent tokens in order:
```
flat_i = t * H_lat * W_lat + h * W_lat + w
```
Scale factors: temporal ×8, spatial ×32 each axis.
IC-LoRA doubles the sequence: `[target tokens | reference tokens]` — only the
first `T*H*W` tokens map to the target output video.

`grid_thw` **must** be the actual latent shape `(T_lat, H_lat, W_lat)`.
For the pipeline to be correct, `height` and `width` must be multiples of 32.
The pipeline enforces this; if you ever pass non-multiples, derive
`grid_thw` from `store.records[0].q_len // 2` and the known T.

## Quick start

```bash
# 1. Check token indices for your prompt
python prompt_token_utils.py \
    --gemma-root /path/to/gemma \
    --prompt "Convert the fountain ..." \
    --token-text fountain

# 2. Discover cross-attention structure (no heatmap output, just a table)
bash launch.sh discover

# 3. Full per-token sweep → sweep.png
bash launch.sh sweep

# 4. Binary mask for "fountain" → mask_fountain.mp4
bash launch.sh mask
```

## Modes

**discover** — prints every `(step, call_idx, q_len, k_len)` observed during
one inference pass.  Look for `k_len ≈ 256` to confirm text cross-attention.
If `k_len` differs, pass `--text-len <N>` to the other modes.

**sweep** — one heatmap per active text token, tiled into `sweep.png`.
Use this to identify which token indices correspond to your word of interest
when `prompt_token_utils.py` cannot find it via tokenization.

**mask** — targeted run for `--word`.  Writes `<work_dir>/mask_<word>.mp4`
with jet-colormap heatmap and contour overlay per frame.

## Offline unit test

```bash
python mask_from_attn.py   # no GPU required; plants a synthetic blob and checks recovery
```
