#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s07_image_only_v0_attn0_img1"
VIDEO_STRENGTH=0.0
CONDITIONING_ATTENTION_STRENGTH=0.0
IMAGE_CONDITION_STRENGTH=1.0

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
