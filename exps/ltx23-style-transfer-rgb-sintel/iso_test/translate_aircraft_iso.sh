#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
TRANSLATE_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/translate_single_sequence.py"

SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/source_video.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel"
SEQUENCE_OUTPUT_DIR="$OUTPUT_ROOT/aircraft_iso"
WORK_DIR="$SEQUENCE_OUTPUT_DIR/intermediates"
OUTPUT_MP4="$SEQUENCE_OUTPUT_DIR/output.mp4"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

DATASET="gtacrime"
FRAME_RATE=24
REFERENCE_VIDEO_CRF=18
SEED=42
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
QUANTIZATION="fp8-cast"

WIDTH="1280"
HEIGHT="768"
NUM_FRAMES=""
SKIP_STAGE_2=1

PROMPT="Follow the motion, timing, camera movement, and scene layout of the source video exactly. Preserve the original composition, shot timing, scene geometry, identity, materials, and as much fine detail as possible while using the RGB control video for structure."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, changed materials, lost details, noisy grain, blurry details, compression artifacts"

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$SEQUENCE_OUTPUT_DIR"

cmd=(
  "$LTX_PYTHON" "$TRANSLATE_SCRIPT"
  --dataset "$DATASET"
  --input-path "$SOURCE_VIDEO"
  --glob "frame_*.png"
  --work-dir "$WORK_DIR"
  --output "$OUTPUT_MP4"
  --reference-video-fps "$FRAME_RATE"
  --reference-video-crf "$REFERENCE_VIDEO_CRF"
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
  --gemma-root "$GEMMA_ROOT"
  --ic-lora-path "$IC_LORA_PATH"
  --conditioning-mode rgb
  --seed "$SEED"
  --frame-rate "$FRAME_RATE"
  --video-strength "$VIDEO_STRENGTH"
  --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
  --no-image-conditioning
  --quiet-layer-streaming
  --quantization "$QUANTIZATION"
)

[[ -n "$WIDTH" ]] && cmd+=(--width "$WIDTH")
[[ -n "$HEIGHT" ]] && cmd+=(--height "$HEIGHT")
[[ -n "$NUM_FRAMES" ]] && cmd+=(--num-frames "$NUM_FRAMES")
[[ "$SKIP_STAGE_2" == "1" ]] && cmd+=(--skip-stage-2)

printf 'Running %s -> %s\n' "$(basename "$SOURCE_VIDEO")" "$OUTPUT_MP4"
printf '%q ' "${cmd[@]}" > "$SEQUENCE_OUTPUT_DIR/command.txt"
printf '\n' >> "$SEQUENCE_OUTPUT_DIR/command.txt"
"${cmd[@]}" 2>&1 | tee "$SEQUENCE_OUTPUT_DIR/run.log"
