#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s09_v1_attn03_img03"
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=0.3
IMAGE_CONDITION_STRENGTH=0.3

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
