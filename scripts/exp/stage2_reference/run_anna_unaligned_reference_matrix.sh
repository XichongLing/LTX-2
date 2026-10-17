#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
PYTHON="$LTX_ROOT/.venv/bin/python3"
RUNNER="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

EXPERIMENT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/stage2_unaligned_reference_only_full129"
SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/source_video.mp4"
APPEARANCE_REFERENCE="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/flux/translated_first_frame.png"
TARGET_MASK="/home/xichong-drz-inokko/experimental/data/disney/masks/anna/source_mask.png"
REFERENCE_MASK="/home/xichong-drz-inokko/experimental/data/disney/masks/anna/reference_mask.png"

DISTILLED_CHECKPOINT="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
CACHE="$EXPERIMENT_ROOT/source_only_stage2_input.safetensors"

PROMPT="Faithfully reconstruct the source video, preserving its motion, timing, camera movement, scene layout, character identity, and geometry."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"
SEED=42
STAGE2_NOISE_SEED=4242
REFERENCE_NOISE_SEED=4245

for required in \
  "$PYTHON" "$RUNNER" "$SOURCE_VIDEO" "$APPEARANCE_REFERENCE" \
  "$TARGET_MASK" "$REFERENCE_MASK" "$DISTILLED_CHECKPOINT" \
  "$SPATIAL_UPSAMPLER" "$GEMMA_ROOT" "$IC_LORA"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required path: %s\n' "$required" >&2
    exit 1
  fi
done

mkdir -p "$EXPERIMENT_ROOT/predictions"
export PYTORCH_ALLOC_CONF=expandable_segments:True

common_args=(
  "$PYTHON" "$RUNNER"
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER"
  --gemma-root "$GEMMA_ROOT"
  --ic-lora-path "$IC_LORA"
  --reference-video "$SOURCE_VIDEO"
  --conditioning-video "$SOURCE_VIDEO"
  --no-image-conditioning
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --seed "$SEED"
  --stage-2-noise-seed "$STAGE2_NOISE_SEED"
  --stage-2-reference-noise-seed "$REFERENCE_NOISE_SEED"
  --width 1280
  --height 768
  --num-frames 129
  --frame-rate 24
  --video-strength 1.0
  --conditioning-mode rgb
  --conditioning-attention-strength 1.0
  --stage-1-ic-lora-strength 1.0
  --stage-2-ic-lora-strength 1.0
  --stage-2-conditioning-attention-strength 1.0
  --first-frame-attention-schedule constant
  --first-frame-attention-multiplier 1.0
  --source-strength-schedule constant
  --source-strength 1.0
  --source-strength-routing symmetric
  --quantization fp8-cast
  --stage-2-branch-mode dual
  --stage-2-routing-mask "$TARGET_MASK"
  --stage-2-dress-video-contribution 0
  --stage-2-dual-image-ic-lora
  --stage-2-kv-mode none
  --stage-2-kv-cache-backend cpu
  --stage-2-reference-edit-conditioning video
  --stage-2-reference-kv-layers all
)

validate_cache() {
  "$PYTHON" - "$CACHE.json" <<'PYCHECK'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"missing Stage-2 cache manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
config = manifest.get("stage_1_config", {})
if config.get("image_conditioning") != []:
    raise SystemExit(f"cache is contaminated by image conditioning: {config.get('image_conditioning')!r}")
print(
    "Validated source-only Stage-2 cache: "
    f"video_checksum={manifest.get('video_checksum')} "
    f"audio_checksum={manifest.get('audio_checksum')}"
)
PYCHECK
}

run_one() {
  local name="$1"
  shift
  local output="$EXPERIMENT_ROOT/${name}.mp4"
  local prediction_dir="$EXPERIMENT_ROOT/predictions/$name"
  local command_file="$EXPERIMENT_ROOT/${name}.command.txt"
  local log_file="$EXPERIMENT_ROOT/${name}.log"
  local cmd=("${common_args[@]}" --output "$output" --stage-2-prediction-dir "$prediction_dir" "$@")

  if [[ -s "$output" && "${OVERWRITE:-0}" != "1" ]]; then
    printf 'Skipping existing output: %s\n' "$output"
    return
  fi
  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  printf 'Running %s -> %s\n' "$name" "$output"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

if [[ ! -s "$CACHE" || ! -s "$CACHE.json" ]]; then
  run_one source_only_baseline \
    --stage-2-reference-attention-mode none \
    --stage-2-reference-kv-strength 0 \
    --save-stage-2-input "$CACHE"
else
  validate_cache
  run_one source_only_baseline \
    --stage-2-reference-attention-mode none \
    --stage-2-reference-kv-strength 0 \
    --load-stage-2-input "$CACHE"
fi
validate_cache

for mode in blend concat; do
  for strength in 0.25 0.5 0.75 1.0; do
    suffix="${strength/./p}"
    run_one "${mode}_s${suffix}" \
      --stage-2-appearance-reference "$APPEARANCE_REFERENCE" \
      --stage-2-appearance-reference-mask "$REFERENCE_MASK" \
      --stage-2-reference-attention-mode "$mode" \
      --stage-2-reference-kv-strength "$strength" \
      --load-stage-2-input "$CACHE"
  done
done

printf 'Completed unaligned-reference matrix in %s\n' "$EXPERIMENT_ROOT"
