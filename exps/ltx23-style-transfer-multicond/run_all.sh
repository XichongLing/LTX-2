#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Sintel multi-conditioning sweep ==="
bash "$SCRIPT_DIR/run_sintel_multicond.sh"

echo "=== Disney previz multi-conditioning sweep ==="
bash "$SCRIPT_DIR/run_disney_previz_multicond.sh"

echo "=== All experiments complete ==="
