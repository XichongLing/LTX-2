# Stage-1 / Stage-2 conditioning wrappers

These wrappers cover the four controlled combinations used in the dual-trajectory
experiments.  They all use the same Stage-1 setup: reference-first-frame image
conditioning, source-video conditioning, and Stage-1 IC-LoRA at strength `1`.

```bash
cd /home/xichong-drz-inokko/experimental/LTX-2

# 1. Stage 1 only
scripts/_dual_traj/01_stage1_image_video.sh SOURCE.mp4 REFERENCE.png RUN_DIR

# 2. Stage 1 + Stage-2 image+video+IC-LoRA (video branch)
scripts/_dual_traj/02_stage1_stage2_image_video.sh SOURCE.mp4 REFERENCE.png RUN_DIR

# 3. Stage 1 + Stage-2 image-only+IC-LoRA (image branch)
scripts/_dual_traj/03_stage1_stage2_image_iclora.sh SOURCE.mp4 REFERENCE.png RUN_DIR

# 4. Stage 1 + Stage-2 dual trajectory with K/V guidance
scripts/_dual_traj/04_stage1_stage2_dual_kv.sh SOURCE.mp4 REFERENCE.png MASK RUN_DIR
```

`MASK` can be a static mask, a mask-video file, a tensor/numpy mask, or a PNG-mask
directory.  The dual wrapper defaults to `partition` K/V guidance, `g=0`, and
runtime IC-LoRA on both Stage-2 trajectories.  For example:

```bash
WIDTH=1280 HEIGHT=768 NUM_FRAMES=81 FRAME_RATE=16 SEED=42 \
DUAL_KV_MODE=append DUAL_G=0.25 \
scripts/_dual_traj/04_stage1_stage2_dual_kv.sh source.mp4 reference.png masks run/dual_append
```

The scripts write `command.txt`, `run.log`, and `output.mp4` to `RUN_DIR`.  The
dual wrapper additionally writes a reusable `stage2_input_cache.safetensors`.
Options after `--` are forwarded to `run_ic_lora_style_transfer.py`; use this to
add probes, prediction dumps, schedules, or an existing `--ref-partition-config`.

For matched Stage-2 comparisons, generate a Stage-2 cache once and invoke the
underlying runner with `--load-stage-2-input`; the standalone wrappers intentionally
remain self-contained rather than silently sharing an input cache.
