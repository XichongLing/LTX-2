#!/usr/bin/env bash
# Stage 1 image+video, then controlled Stage 2 video branch (image+video+IC-LoRA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

parse_common_args "$@" || exit $?
build_common_command

run_and_record \
  "${COMMON_COMMAND[@]}" \
  --stage-2-branch-mode video \
  --stage-2-lora-execution runtime \
  --stage-2-ic-lora-strength 1.0 \
  --stage-2-conditioning-attention-strength 1.0 \
  --output "${OUTPUT_DIR}/output.mp4" \
  "${EXTRA_LTX_ARGS[@]}"
