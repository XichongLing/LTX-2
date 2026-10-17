#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s02_image_high_v1_attn1_img1"
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
IMAGE_CONDITION_STRENGTH=1.0

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
