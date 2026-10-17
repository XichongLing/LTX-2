#!/bin/bash

set -euo pipefail

EXPERIMENT_NAME="s01_baseline_v1_attn1_img07"
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
IMAGE_CONDITION_STRENGTH=0.7

source "/home/xichong-drz-inokko/experimental/LTX-2/scripts/exp/upscaling/strength_experiments/vgray_strength_common.sh"
