#!/usr/bin/env bash
# Stage 1 only: reference-first-frame image + source-video conditioning + IC-LoRA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

parse_common_args "$@" || exit $?
build_common_command

run_and_record \
  "${COMMON_COMMAND[@]}" \
  --stage-2-ic-lora-strength 0.0 \
  --skip-stage-2 \
  --output "${OUTPUT_DIR}/output.mp4" \
  "${EXTRA_LTX_ARGS[@]}"
