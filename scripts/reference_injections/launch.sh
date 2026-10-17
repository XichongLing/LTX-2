#!/usr/bin/env bash
# Launch the reference-injection segmentation check.
# Edit the variables below, then pick a MODE (discover / sweep / mask).

set -euo pipefail

REPO=/home/xichong-drz-inokko/experimental/LTX-2
PYTHON=$REPO/.venv/bin/python3

REFERENCE_VIDEO=/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/fountain_real/output.mp4
WORK_DIR=/home/xichong-drz-inokko/experimental/Image_seq_transl/local_edit/segcheck/remive

DISTILLED=$REPO/models/ltx-2.3-22b-distilled-1.1.safetensors
UPSAMPLER=$REPO/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
GEMMA=/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized
ICLORA=$REPO/models/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors

PROMPT="Convert the fountain into a low-poly, proxy-mesh-alike model, while following the original motions, scene structures, and camera poses of the source video."
WORD="fountain"

# MODE: discover | sweep | mask
MODE=${1:-discover}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON $REPO/scripts/reference_injections/run_reference_mask_check.py "$MODE" \
    --reference-video  "$REFERENCE_VIDEO" \
    --prompt           "$PROMPT" \
    --gemma-root       "$GEMMA" \
    --distilled-checkpoint-path "$DISTILLED" \
    --spatial-upsampler-path    "$UPSAMPLER" \
    --ic-lora-path     "$ICLORA" \
    --height   768 \
    --width    1280 \
    --num-frames 97 \
    --video-cfg-scale 2.0 \
    --video-strength  0.85 \
    --work-dir  "$WORK_DIR" \
    --word      "$WORD" \
    --text-len  1024 \
    --quantization fp8-cast

echo "Done. Outputs in $WORK_DIR"
