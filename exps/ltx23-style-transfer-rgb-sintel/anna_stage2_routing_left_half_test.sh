#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="${LTX_ROOT:-/home/xichong-drz-inokko/experimental/LTX-2}"
LTX_PYTHON="${LTX_PYTHON:-$LTX_ROOT/.venv/bin/python3}"
RUN_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

INPUT_ROOT="${INPUT_ROOT:-/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond}"
INTERMEDIATES_DIR="$INPUT_ROOT/intermediates"
SOURCE_VIDEO="${SOURCE_VIDEO:-$INTERMEDIATES_DIR/source_video.mp4}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-$INTERMEDIATES_DIR/flux/translated_first_frame.png}"
ROUTING_MASK="${ROUTING_MASK:-$INTERMEDIATES_DIR/stage2_left_half_mask.png}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$INPUT_ROOT/stage2_routing_left_half_test}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$OUTPUT_ROOT/predictions}"
STAGE_2_INPUT_CACHE="${STAGE_2_INPUT_CACHE:-$OUTPUT_ROOT/anna_stage2_input.safetensors}"

DISTILLED_CHECKPOINT_PATH="${DISTILLED_CHECKPOINT_PATH:-$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors}"
SPATIAL_UPSAMPLER_PATH="${SPATIAL_UPSAMPLER_PATH:-$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized}"
IC_LORA_PATH="${IC_LORA_PATH:-$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors}"

FRAME_RATE="${FRAME_RATE:-24}"
SEED="${SEED:-42}"
STAGE_2_NOISE_SEED="${STAGE_2_NOISE_SEED:-4242}"
NUM_FRAMES="${NUM_FRAMES:-129}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-768}"

IMAGE_STRENGTH="${IMAGE_STRENGTH:-1.0}"
VIDEO_STRENGTH="${VIDEO_STRENGTH:-1.0}"
CONDITIONING_ATTENTION_STRENGTH="${CONDITIONING_ATTENTION_STRENGTH:-1.0}"
STAGE_2_IC_LORA_STRENGTH="${STAGE_2_IC_LORA_STRENGTH:-1.0}"
STAGE_2_CONDITIONING_ATTENTION_STRENGTH="${STAGE_2_CONDITIONING_ATTENTION_STRENGTH:-1.0}"
STAGE_2_VIDEO_MIX="${STAGE_2_VIDEO_MIX:-0.5}"
RUN_SIGMA_RAMP="${RUN_SIGMA_RAMP:-1}"

PROMPT="${PROMPT:-Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts}"

for required_path in \
  "$LTX_PYTHON" \
  "$RUN_SCRIPT" \
  "$SOURCE_VIDEO" \
  "$REFERENCE_IMAGE" \
  "$ROUTING_MASK" \
  "$DISTILLED_CHECKPOINT_PATH" \
  "$SPATIAL_UPSAMPLER_PATH" \
  "$GEMMA_ROOT" \
  "$IC_LORA_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'Missing required path: %s\n' "$required_path" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTPUT_ROOT" "$PREDICTION_ROOT"

BASE_ARGS=(
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
  --gemma-root "$GEMMA_ROOT"
  --ic-lora-path "$IC_LORA_PATH"
  --reference-video "$SOURCE_VIDEO"
  --conditioning-video "$SOURCE_VIDEO"
  --reference-image "$REFERENCE_IMAGE"
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --seed "$SEED"
  --stage-2-noise-seed "$STAGE_2_NOISE_SEED"
  --width "$WIDTH"
  --height "$HEIGHT"
  --num-frames "$NUM_FRAMES"
  --frame-rate "$FRAME_RATE"
  --image-strength "$IMAGE_STRENGTH"
  --image-crf 33
  --video-strength "$VIDEO_STRENGTH"
  --conditioning-mode rgb
  --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
  --stage-2-ic-lora-strength "$STAGE_2_IC_LORA_STRENGTH"
  --stage-2-conditioning-attention-strength "$STAGE_2_CONDITIONING_ATTENTION_STRENGTH"
  --first-frame-attention-schedule constant
  --first-frame-attention-multiplier 1.0
  --source-strength-routing symmetric
  --quiet-layer-streaming
  --quantization fp8-cast
)

run_case() {
  local name="$1"
  shift
  local output_video="$OUTPUT_ROOT/${name}.mp4"
  local prediction_dir="$PREDICTION_ROOT/$name"
  local command_file="$OUTPUT_ROOT/${name}.command.txt"
  local log_file="$OUTPUT_ROOT/${name}.log"
  local cmd=(
    "$LTX_PYTHON" "$RUN_SCRIPT"
    "${BASE_ARGS[@]}"
    --output "$output_video"
    --stage-2-prediction-dir "$prediction_dir"
    "$@"
  )

  printf 'Running %s -> %s\n' "$name" "$output_video"
  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

run_case image_cache \
  --stage-2-branch-mode image \
  --source-strength-schedule constant \
  --source-strength 1.0 \
  --save-stage-2-input "$STAGE_2_INPUT_CACHE"

run_case video_constant \
  --stage-2-branch-mode video \
  --source-strength-schedule constant \
  --source-strength 1.0 \
  --load-stage-2-input "$STAGE_2_INPUT_CACHE"

run_case global_mix_${STAGE_2_VIDEO_MIX/./p} \
  --stage-2-branch-mode global \
  --stage-2-video-mix "$STAGE_2_VIDEO_MIX" \
  --source-strength-schedule constant \
  --source-strength 1.0 \
  --load-stage-2-input "$STAGE_2_INPUT_CACHE"

if [[ "$RUN_SIGMA_RAMP" == "1" ]]; then
  run_case video_sigma_ramp \
    --stage-2-branch-mode video \
    --source-strength-schedule sigma-ramp \
    --source-strength-early 1.0 \
    --source-strength-late 0.30 \
    --source-strength-fade-start-sigma 0.725 \
    --source-strength-fade-end-sigma 0.421875 \
    --load-stage-2-input "$STAGE_2_INPUT_CACHE"
else
  printf 'Skipping video_sigma_ramp because RUN_SIGMA_RAMP=%s\n' "$RUN_SIGMA_RAMP"
fi

for dress_video_contribution in 0 0.25 0.5; do
  case_name="spatial_g${dress_video_contribution/./p}"
  run_case "$case_name" \
    --stage-2-branch-mode spatial \
    --stage-2-routing-mask "$ROUTING_MASK" \
    --stage-2-dress-video-contribution "$dress_video_contribution" \
    --source-strength-schedule constant \
    --source-strength 1.0 \
    --load-stage-2-input "$STAGE_2_INPUT_CACHE"
done

printf 'Stage-2 routing test completed. Outputs: %s\n' "$OUTPUT_ROOT"
