#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
TRANSLATE_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/translate_single_sequence.py"

SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_dress/intermediates/source_video.mp4"
REFERENCE_IMAGE="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_dress/intermediates/flux/translated_first_frame.png"
OUTPUT_DIR="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond"
WORK_DIR="$OUTPUT_DIR/intermediates"
OUTPUT_VIDEO="$OUTPUT_DIR/output.mp4"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

FRAME_RATE=24
SEED=42
WIDTH=1280
HEIGHT=768
IMAGE_STRENGTH=1.0
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
STAGE_2_IC_LORA_STRENGTH=1.0
STAGE_2_CONDITIONING_ATTENTION_STRENGTH=1.0

PROMPT="Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"

for required_path in \
  "$LTX_PYTHON" \
  "$TRANSLATE_SCRIPT" \
  "$SOURCE_VIDEO" \
  "$REFERENCE_IMAGE" \
  "$DISTILLED_CHECKPOINT_PATH" \
  "$SPATIAL_UPSAMPLER_PATH" \
  "$GEMMA_ROOT" \
  "$IC_LORA_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'Missing required path: %s\n' "$required_path" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

cmd=(
  "$LTX_PYTHON" "$TRANSLATE_SCRIPT"
  --dataset gtacrime
  --input-path "$SOURCE_VIDEO"
  --translated-first-frame "$REFERENCE_IMAGE"
  --work-dir "$WORK_DIR"
  --output "$OUTPUT_VIDEO"
  --reference-video-fps "$FRAME_RATE"
  --reference-video-crf 18
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
  --gemma-root "$GEMMA_ROOT"
  --ic-lora-path "$IC_LORA_PATH"
  --conditioning-mode rgb
  --seed "$SEED"
  --width "$WIDTH"
  --height "$HEIGHT"
  --frame-rate "$FRAME_RATE"
  --image-strength "$IMAGE_STRENGTH"
  --image-crf 33
  --video-strength "$VIDEO_STRENGTH"
  --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
  --stage-2-ic-lora-strength "$STAGE_2_IC_LORA_STRENGTH"
  --stage-2-conditioning-attention-strength "$STAGE_2_CONDITIONING_ATTENTION_STRENGTH"
  --quiet-layer-streaming
  --quantization fp8-cast
)

printf 'Running Anna stage-2 video-conditioned style transfer -> %s\n' "$OUTPUT_VIDEO"
printf '%q ' "${cmd[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
