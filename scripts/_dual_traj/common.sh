#!/usr/bin/env bash
# Shared configuration for the small, reproducible Stage-1/Stage-2 experiment
# launchers in this directory. Source this file; do not invoke it directly.

set -euo pipefail

DUAL_TRAJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX_REPO="${LTX_REPO:-$(cd "${DUAL_TRAJ_DIR}/../.." && pwd)}"
LTX_PYTHON="${LTX_PYTHON:-${LTX_REPO}/.venv/bin/python3}"
LTX_RUNNER="${LTX_RUNNER:-${LTX_REPO}/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py}"

# These are the local standard checkpoints. Override any of them in the shell
# when using another model installation.
DISTILLED_CHECKPOINT="${DISTILLED_CHECKPOINT:-${LTX_REPO}/models/ltx-2.3-22b-distilled-1.1.safetensors}"
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-${LTX_REPO}/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized}"
IC_LORA="${IC_LORA:-${LTX_REPO}/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors}"

SEED="${SEED:-42}"
STAGE2_NOISE_SEED="${STAGE2_NOISE_SEED:-${SEED}}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
REFERENCE_DOWNSCALE_FACTOR="${REFERENCE_DOWNSCALE_FACTOR:-2}"

PROMPT="${PROMPT:-Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts}"

usage_common() {
  cat <<'EOF'
Common positional arguments:
  SOURCE_VIDEO REFERENCE_IMAGE OUTPUT_DIR [-- EXTRA_LTX_ARGUMENTS...]

Useful environment overrides:
  SEED=42 STAGE2_NOISE_SEED=42 WIDTH=1280 HEIGHT=768 NUM_FRAMES=81 FRAME_RATE=16
  PROMPT='...' NEGATIVE_PROMPT='...' QUANTIZATION=fp8-cast
  LTX_REPO=... LTX_PYTHON=... DISTILLED_CHECKPOINT=... SPATIAL_UPSAMPLER=...
  GEMMA_ROOT=... IC_LORA=... REFERENCE_DOWNSCALE_FACTOR=2

Arguments after -- are appended verbatim to run_ic_lora_style_transfer.py.
EOF
}

parse_common_args() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then
    usage_common
    return 1
  fi
  SOURCE_VIDEO="$1"
  REFERENCE_IMAGE="$2"
  OUTPUT_DIR="$3"
  shift 3
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  EXTRA_LTX_ARGS=("$@")

  [[ -f "$SOURCE_VIDEO" ]] || { echo "Missing source video: $SOURCE_VIDEO" >&2; return 2; }
  [[ -f "$REFERENCE_IMAGE" ]] || { echo "Missing reference image: $REFERENCE_IMAGE" >&2; return 2; }
  [[ -x "$LTX_PYTHON" ]] || { echo "Missing Python executable: $LTX_PYTHON" >&2; return 2; }
  [[ -f "$LTX_RUNNER" ]] || { echo "Missing runner: $LTX_RUNNER" >&2; return 2; }
  mkdir -p "$OUTPUT_DIR"
}

append_optional_dimensions() {
  COMMON_DIMENSION_ARGS=()
  [[ -n "${WIDTH:-}" ]] && COMMON_DIMENSION_ARGS+=(--width "$WIDTH")
  [[ -n "${HEIGHT:-}" ]] && COMMON_DIMENSION_ARGS+=(--height "$HEIGHT")
  [[ -n "${NUM_FRAMES:-}" ]] && COMMON_DIMENSION_ARGS+=(--num-frames "$NUM_FRAMES")
  [[ -n "${FRAME_RATE:-}" ]] && COMMON_DIMENSION_ARGS+=(--frame-rate "$FRAME_RATE")
  # A missing optional dimension must not make this helper fail under `set -e`.
  return 0
}

build_common_command() {
  append_optional_dimensions
  COMMON_COMMAND=(
    "$LTX_PYTHON" "$LTX_RUNNER"
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT"
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER"
    --gemma-root "$GEMMA_ROOT"
    --ic-lora-path "$IC_LORA"
    --reference-video "$SOURCE_VIDEO"
    --conditioning-video "$SOURCE_VIDEO"
    --reference-image "$REFERENCE_IMAGE"
    --prompt "$PROMPT"
    --negative-prompt "$NEGATIVE_PROMPT"
    --seed "$SEED"
    --stage-2-noise-seed "$STAGE2_NOISE_SEED"
    "${COMMON_DIMENSION_ARGS[@]}"
    --image-strength 1.0
    --image-crf 33
    --video-strength 1.0
    --conditioning-mode rgb
    --conditioning-attention-strength 1.0
    --stage-1-ic-lora-strength 1.0
    --first-frame-attention-schedule constant
    --first-frame-attention-multiplier 1.0
    --source-strength-schedule constant
    --source-strength 1.0
    --source-strength-routing target-queries-only
    --reference-downscale-factor-override "$REFERENCE_DOWNSCALE_FACTOR"
    --quantization "$QUANTIZATION"
  )
}

run_and_record() {
  local -a command=("$@")
  printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/command.txt"
  printf '\n' >> "${OUTPUT_DIR}/command.txt"
  (
    cd "$LTX_REPO"
    "${command[@]}"
  ) 2>&1 | tee "${OUTPUT_DIR}/run.log"
}
