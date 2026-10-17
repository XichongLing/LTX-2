#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s06_image_dominant_v04_attn04_img1"
VIDEO_STRENGTH=0.4
CONDITIONING_ATTENTION_STRENGTH=0.4
IMAGE_CONDITION_STRENGTH=1.0

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
