#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s05_attention_weak_v1_attn03_img07"
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=0.3
IMAGE_CONDITION_STRENGTH=0.7

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
