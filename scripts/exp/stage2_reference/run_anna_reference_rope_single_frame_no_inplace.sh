#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
PYTHON="$LTX_ROOT/.venv/bin/python3"
RUNNER="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
BASE_OUTPUT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond"
SOURCE_VIDEO="$BASE_OUTPUT/stage2_reference_rope_single_frame/source_first_frame.mp4"
SOURCE_FIRST_FRAME="$BASE_OUTPUT/intermediates/source_first_frame.png"
APPEARANCE_REFERENCE="$BASE_OUTPUT/intermediates/flux/translated_first_frame.png"
TARGET_MASK="/home/xichong-drz-inokko/experimental/data/disney/masks/anna/source_mask.png"
REFERENCE_MASK="/home/xichong-drz-inokko/experimental/data/disney/masks/anna/reference_mask.png"

DISTILLED_CHECKPOINT="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

PROMPT="Faithfully reconstruct the source video, preserving its motion, timing, camera movement, scene layout, character identity, and geometry."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"
SEED=42
STAGE2_NOISE_SEED=4242
REFERENCE_NOISE_SEED=4245

for required in \
  "$PYTHON" "$RUNNER" "$SOURCE_VIDEO" "$SOURCE_FIRST_FRAME" "$APPEARANCE_REFERENCE" \
  "$TARGET_MASK" "$REFERENCE_MASK" "$DISTILLED_CHECKPOINT" \
  "$SPATIAL_UPSAMPLER" "$GEMMA_ROOT" "$IC_LORA"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required path: %s\n' "$required" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF=expandable_segments:True

run_matrix() {
  local variant="$1"
  local stage1_lora="$2"
  local stage2_lora="$3"
  local image_mode="$4"
  local experiment_root="$BASE_OUTPUT/$variant"
  local cache="$experiment_root/stage2_input.safetensors"
  mkdir -p "$experiment_root/predictions"

  local common_args=(
    "$PYTHON" "$RUNNER"
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT"
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER"
    --gemma-root "$GEMMA_ROOT"
    --ic-lora-path "$IC_LORA"
    --reference-video "$SOURCE_VIDEO"
    --conditioning-video "$SOURCE_VIDEO"
    --prompt "$PROMPT"
    --negative-prompt "$NEGATIVE_PROMPT"
    --seed "$SEED"
    --stage-2-noise-seed "$STAGE2_NOISE_SEED"
    --stage-2-reference-noise-seed "$REFERENCE_NOISE_SEED"
    --width 1280
    --height 768
    --num-frames 1
    --frame-rate 24
    --video-strength 1.0
    --conditioning-mode rgb
    --conditioning-attention-strength 1.0
    --stage-1-ic-lora-strength "$stage1_lora"
    --stage-2-ic-lora-strength "$stage2_lora"
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
  if [[ "$image_mode" == "none" ]]; then
    common_args+=(--no-image-conditioning)
  elif [[ "$image_mode" == "source_first_frame" ]]; then
    common_args+=(--reference-image "$SOURCE_FIRST_FRAME" --image-strength 1.0 --image-crf 33)
  else
    printf 'Unknown image mode: %s\n' "$image_mode" >&2
    exit 1
  fi

  validate_cache() {
    "$PYTHON" - "$cache.json" "$image_mode" "$stage1_lora" "$SOURCE_FIRST_FRAME" <<'PYCHECK'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
image_mode = sys.argv[2]
expected_lora = float(sys.argv[3])
source_first_frame = str(Path(sys.argv[4]).resolve())
if not manifest_path.is_file():
    raise SystemExit(f"missing cache manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
config = manifest.get("stage_1_config", {})
if float(config.get("stage_1_ic_lora_strength", -1)) != expected_lora:
    raise SystemExit("cache Stage-1 IC-LoRA strength mismatch")
images = config.get("image_conditioning")
if image_mode == "none" and images != []:
    raise SystemExit(f"expected no image conditioning, found {images!r}")
if image_mode == "source_first_frame":
    if not isinstance(images, list) or len(images) != 1:
        raise SystemExit(f"expected one source-first-frame condition, found {images!r}")
    if images[0].get("asset", {}).get("path") != source_first_frame or images[0].get("frame_idx") != 0:
        raise SystemExit(f"wrong frame-0 image condition: {images[0]!r}")
print(
    f"Validated {image_mode} cache: fingerprint={manifest.get('stage_1_fingerprint')} "
    f"video_checksum={manifest.get('video_checksum')}"
)
PYCHECK
  }

  run_one() {
    local name="$1"
    shift
    local output="$experiment_root/${name}.mp4"
    local command_file="$experiment_root/${name}.command.txt"
    local log_file="$experiment_root/${name}.log"
    local prediction_dir="$experiment_root/predictions/$name"
    local cmd=("${common_args[@]}" --output "$output" --stage-2-prediction-dir "$prediction_dir" "$@")
    if [[ -s "$output" && "${OVERWRITE:-0}" != "1" ]]; then
      printf 'Skipping existing output: %s\n' "$output"
      return
    fi
    printf '%q ' "${cmd[@]}" > "$command_file"
    printf '\n' >> "$command_file"
    printf 'Running %s/%s -> %s\n' "$variant" "$name" "$output"
    "${cmd[@]}" 2>&1 | tee "$log_file"
  }

  if [[ ! -s "$cache" || ! -s "$cache.json" ]]; then
    run_one source_only_baseline \
      --stage-2-reference-attention-mode none \
      --stage-2-reference-kv-strength 0 \
      --save-stage-2-input "$cache"
  else
    validate_cache
    run_one source_only_baseline \
      --stage-2-reference-attention-mode none \
      --stage-2-reference-kv-strength 0 \
      --load-stage-2-input "$cache"
  fi
  validate_cache

  for position_mode in pre-rope post-rope; do
    local position_suffix="${position_mode/-rope/}"
    run_one "concat_${position_suffix}_rope_s1p0" \
      --stage-2-appearance-reference "$APPEARANCE_REFERENCE" \
      --stage-2-appearance-reference-mask "$REFERENCE_MASK" \
      --stage-2-reference-attention-mode concat \
      --stage-2-reference-position-mode "$position_mode" \
      --stage-2-reference-kv-strength 1.0 \
      --load-stage-2-input "$cache"
  done
}

run_matrix "stage2_reference_rope_single_frame_no_inplace" 1 1 none
printf 'Completed single-frame image-style pre/post-RoPE reference experiment without in-place replacement.\n'
