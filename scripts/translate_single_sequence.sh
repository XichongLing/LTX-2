#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl"
DATA_ROOT="/home/xichong-drz-inokko/experimental/data/MPI-Sintel-training_images/training/albedo"

DATA_DIR="$DATA_ROOT/ambush_5"
ATTR="pr"

SEQ_NAME="${DATA_DIR##*/}"
OUTPUT_NAME="${SEQ_NAME}_${ATTR}"
RUN_ROOT="$OUTPUT_ROOT/${OUTPUT_NAME}"
INTERMEDIATE_ROOT="$RUN_ROOT/intermediates"
FLUX_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/flux"
CONTROL_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/controls"
OUT_MP4="$RUN_ROOT/output_lprompt.mp4"
DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/depth.mp4"

mkdir -p "$RUN_ROOT" "$FLUX_INTERMEDIATE_DIR" "$CONTROL_INTERMEDIATE_DIR"


export PYTORCH_ALLOC_CONF=expandable_segments:True

"$LTX_PYTHON" "$LTX_ROOT/scripts/run_sequence_ic_lora_pipeline.py" \
  --input-dir "$DATA_DIR" \
  --glob "frame_*.png" \
  --output "$OUT_MP4" \
  --work-dir "$INTERMEDIATE_ROOT" \
  --reference-video-fps 24 \
  --prompt "Follow the motion and camera of the source sequence, but render it in the look, materials, lighting, and style implied by the translated first frame, make objects and the environments look like photorealistic ones in a live-action movie." \
  --use-local-flux-runner \
  --flux-python "$FLUX_PYTHON" \
  --flux-runner-path "$FLUX_ROOT/run_flux_first_frame_style_transfer.py" \
  --flux-output-name "flux/translated_first_frame.png" \
  --flux-prompt "Convert this rough render into a polished cinematic final render with sophisticated materials, lighting, shadows, textures, and realism while preserving the original composition and geometry." \
  --distilled-checkpoint-path /home/xichong-drz-inokko/Downloads/ltx-2-19b-distilled.safetensors \
  --spatial-upsampler-path /home/xichong-drz-inokko/experimental/ComfyUI/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized \
  --ic-lora-path /home/xichong-drz-inokko/experimental/ComfyUI/models/loras/LTX-2/ltx-2-19b-ic-lora-depth-control.safetensors \
  --conditioning-mode depth \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --depth-output "$DEPTH_MP4" \
  --width 1280 \
  --height 704 \
  --image-strength 1.0 \
  --video-strength 1.0 \
  --conditioning-attention-strength 1.0 \
  --quiet-layer-streaming \
  --quantization fp8-cast \
  --skip-stage-2
