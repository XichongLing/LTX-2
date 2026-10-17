#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/s01_baseline_v1_attn1_img07.sh"
bash "$SCRIPT_DIR/s02_image_high_v1_attn1_img1.sh"
bash "$SCRIPT_DIR/s03_image_off_v1_attn1_img0.sh"
bash "$SCRIPT_DIR/s04_video_weak_v03_attn1_img07.sh"
bash "$SCRIPT_DIR/s05_attention_weak_v1_attn03_img07.sh"
bash "$SCRIPT_DIR/s06_image_dominant_v04_attn04_img1.sh"
bash "$SCRIPT_DIR/s07_image_only_v0_attn0_img1.sh"
