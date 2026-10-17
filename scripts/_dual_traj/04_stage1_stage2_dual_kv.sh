#!/usr/bin/env bash
# Stage 1 image+video, then Stage 2 independent video/edit trajectories with
# video-anchor K/V guidance in the edit region.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 4 ]]; then
  cat <<'EOF'
Usage:
  04_stage1_stage2_dual_kv.sh SOURCE_VIDEO REFERENCE_IMAGE ROUTING_MASK OUTPUT_DIR [-- EXTRA_LTX_ARGUMENTS...]

ROUTING_MASK may be a mask PNG, video, .npy/.pt tensor, or a directory of mask PNGs.
Defaults: partition K/V guidance, strength 1.0, g=0.0, CPU K/V cache, and runtime
IC-LoRA on both the anchor and editable/image trajectory.
Override via DUAL_KV_MODE, DUAL_KV_STRENGTH, DUAL_G, DUAL_KV_LAYERS, or by appending
explicit runner arguments after --.
EOF
  exit 1
fi

SOURCE_VIDEO="$1"
REFERENCE_IMAGE="$2"
ROUTING_MASK="$3"
OUTPUT_DIR="$4"
shift 4
if [[ "${1:-}" == "--" ]]; then shift; fi
parse_common_args "$SOURCE_VIDEO" "$REFERENCE_IMAGE" "$OUTPUT_DIR" "$@" || exit $?
[[ -e "$ROUTING_MASK" ]] || { echo "Missing routing mask: $ROUTING_MASK" >&2; exit 2; }

DUAL_KV_MODE="${DUAL_KV_MODE:-partition}"
DUAL_KV_STRENGTH="${DUAL_KV_STRENGTH:-1.0}"
DUAL_G="${DUAL_G:-0.0}"
DUAL_KV_LAYERS="${DUAL_KV_LAYERS:-all}"
DUAL_KV_CACHE_BACKEND="${DUAL_KV_CACHE_BACKEND:-cpu}"

build_common_command
run_and_record \
  "${COMMON_COMMAND[@]}" \
  --stage-2-branch-mode dual \
  --stage-2-lora-execution runtime \
  --stage-2-ic-lora-strength 1.0 \
  --stage-2-conditioning-attention-strength 1.0 \
  --stage-2-dual-image-ic-lora \
  --stage-2-kv-mode "$DUAL_KV_MODE" \
  --stage-2-kv-strength "$DUAL_KV_STRENGTH" \
  --stage-2-kv-layers "$DUAL_KV_LAYERS" \
  --stage-2-kv-cache-backend "$DUAL_KV_CACHE_BACKEND" \
  --stage-2-routing-mask "$ROUTING_MASK" \
  --stage-2-dress-video-contribution "$DUAL_G" \
  --save-stage-2-input "${OUTPUT_DIR}/stage2_input_cache.safetensors" \
  --output "${OUTPUT_DIR}/output.mp4" \
  "${EXTRA_LTX_ARGS[@]}"
