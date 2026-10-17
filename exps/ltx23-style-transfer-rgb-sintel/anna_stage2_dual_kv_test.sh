#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="${LTX_ROOT:-/home/xichong-drz-inokko/experimental/LTX-2}"
LTX_PYTHON="${LTX_PYTHON:-$LTX_ROOT/.venv/bin/python3}"
RUN_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

INPUT_ROOT="${INPUT_ROOT:-/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond}"
SOURCE_VIDEO="${SOURCE_VIDEO:-$INPUT_ROOT/intermediates/source_video.mp4}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-$INPUT_ROOT/intermediates/flux/translated_first_frame.png}"
ROUTING_MASK="${ROUTING_MASK:-$INPUT_ROOT/intermediates/stage2_left_half_mask.png}"

NUM_FRAMES="${NUM_FRAMES:-17}"
KV_LAYERS="${KV_LAYERS:-8,16,24,32,40,47}"
FRAME_RATE="${FRAME_RATE:-24}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-768}"
SEED="${SEED:-42}"
STAGE_2_NOISE_SEED="${STAGE_2_NOISE_SEED:-4242}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$INPUT_ROOT/stage2_dual_kv_f${NUM_FRAMES}}"
PREDICTION_ROOT="${PREDICTION_ROOT:-$OUTPUT_ROOT/predictions}"
STAGE_2_INPUT_CACHE="${STAGE_2_INPUT_CACHE:-$OUTPUT_ROOT/anna_stage2_input.safetensors}"
KV_CACHE_BACKEND="${KV_CACHE_BACKEND:-cpu}"
KV_CACHE_DIR="${KV_CACHE_DIR:-$OUTPUT_ROOT/kv_cache}"
G_VALUES="${G_VALUES:-0 0.1 0.25}"
RUN_BASELINES="${RUN_BASELINES:-1}"

DISTILLED_CHECKPOINT_PATH="${DISTILLED_CHECKPOINT_PATH:-$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors}"
SPATIAL_UPSAMPLER_PATH="${SPATIAL_UPSAMPLER_PATH:-$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized}"
IC_LORA_PATH="${IC_LORA_PATH:-$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors}"

PROMPT="${PROMPT:-Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts}"

REQUIRED_PATHS=(
  "$LTX_PYTHON"
  "$RUN_SCRIPT"
  "$SOURCE_VIDEO"
  "$REFERENCE_IMAGE"
  "$ROUTING_MASK"
  "$DISTILLED_CHECKPOINT_PATH"
  "$SPATIAL_UPSAMPLER_PATH"
  "$GEMMA_ROOT"
  "$IC_LORA_PATH"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'Missing required path: %s\n' "$required_path" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTPUT_ROOT" "$PREDICTION_ROOT"
if [[ "$KV_CACHE_BACKEND" == "disk" ]]; then
  mkdir -p "$KV_CACHE_DIR"
fi

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
  --image-strength 1.0
  --image-crf 33
  --video-strength 1.0
  --conditioning-mode rgb
  --conditioning-attention-strength 1.0
  --stage-2-ic-lora-strength 1.0
  --stage-2-conditioning-attention-strength 1.0
  --first-frame-attention-schedule constant
  --first-frame-attention-multiplier 1.0
  --source-strength-schedule constant
  --source-strength 1.0
  --source-strength-routing symmetric
  --quantization fp8-cast
)

run_case() {
  local name="$1"
  shift
  local output_video="$OUTPUT_ROOT/${name}.mp4"
  local command_file="$OUTPUT_ROOT/${name}.command.txt"
  local log_file="$OUTPUT_ROOT/${name}.log"
  local cmd=(
    "$LTX_PYTHON" "$RUN_SCRIPT"
    "${BASE_ARGS[@]}"
    --output "$output_video"
    --stage-2-prediction-dir "$PREDICTION_ROOT/$name"
    "$@"
  )
  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  printf 'Running %s -> %s\n' "$name" "$output_video"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

SHARED_ARGS=(
  --stage-2-branch-mode spatial
  --stage-2-routing-mask "$ROUTING_MASK"
  --stage-2-dress-video-contribution 0
  --save-stage-2-input "$STAGE_2_INPUT_CACHE"
)
if [[ "$RUN_BASELINES" == "1" ]]; then
  run_case shared_spatial_g0 "${SHARED_ARGS[@]}"
fi

KV_CACHE_ARGS=(--stage-2-kv-cache-backend "$KV_CACHE_BACKEND")
if [[ "$KV_CACHE_BACKEND" == "disk" ]]; then
  KV_CACHE_ARGS+=(--stage-2-kv-cache-dir "$KV_CACHE_DIR")
fi

DUAL_NONE_ARGS=(
  --stage-2-branch-mode dual
  --stage-2-routing-mask "$ROUTING_MASK"
  --stage-2-dress-video-contribution 0
  --stage-2-kv-mode none
  --stage-2-kv-layers "$KV_LAYERS"
  "${KV_CACHE_ARGS[@]}"
  --load-stage-2-input "$STAGE_2_INPUT_CACHE"
)
if [[ "$RUN_BASELINES" == "1" ]]; then
  run_case dual_none_g0 "${DUAL_NONE_ARGS[@]}"
fi

for kv_mode in partition append; do
  for g_dress in $G_VALUES; do
    case_name="dual_${kv_mode}_g${g_dress/./p}"
    CASE_ARGS=(
      --stage-2-branch-mode dual
      --stage-2-routing-mask "$ROUTING_MASK"
      --stage-2-dress-video-contribution "$g_dress"
      --stage-2-kv-mode "$kv_mode"
      --stage-2-kv-strength 1.0
      --stage-2-kv-layers "$KV_LAYERS"
      "${KV_CACHE_ARGS[@]}"
      --load-stage-2-input "$STAGE_2_INPUT_CACHE"
    )
    run_case "$case_name" "${CASE_ARGS[@]}"
  done
done

printf 'Dual-trajectory KV ablations completed: %s\n' "$OUTPUT_ROOT"
printf 'For the full run use NUM_FRAMES=129 KV_LAYERS=all OUTPUT_ROOT=<path> %s\n' "$0"
