#!/usr/bin/env bash
# Stage 1 image+video, then controlled Stage 2 image branch with runtime IC-LoRA.
# Source-video conditioning remains registered for Stage 1, but is absent from the
# Stage-2 image-branch transformer evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

parse_common_args "$@" || exit $?
build_common_command

run_and_record \
  "${COMMON_COMMAND[@]}" \
  --stage-2-branch-mode image \
  --stage-2-lora-execution runtime \
  --stage-2-ic-lora-strength 1.0 \
  --output "${OUTPUT_DIR}/output.mp4" \
  "${EXTRA_LTX_ARGS[@]}"
