#!/usr/bin/env bash
set -euo pipefail
cd /home/xichong-drz-inokko/experimental/LTX-2
SRC=/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/source_video.mp4
BASE=/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs
PY=/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python3
for PARAM in velocity x0; do
  for DTYPE in fp64 bf16; do
    OUT="$BASE/path_compare_plane_10steps_${PARAM}_${DTYPE}"
    LOG="$BASE/path_compare_plane_10steps_${PARAM}_${DTYPE}.log"
    mkdir -p "$OUT"
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START param=$PARAM dtype=$DTYPE out=$OUT"
    PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" scripts/inversion/run_rf_inversion.py       --source-video "$SRC"       --method contextflow-rf-solver-2       --num-steps 10       --use-tiled-encode       --contextflow-step-parameterization "$PARAM"       --contextflow-step-compute-dtype "$DTYPE"       --output-dir "$OUT" > "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE param=$PARAM dtype=$DTYPE log=$LOG"
  done
done
