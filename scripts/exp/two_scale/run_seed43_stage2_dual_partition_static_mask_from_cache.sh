#!/usr/bin/env bash
set -euo pipefail

REPO=/home/xichong-drz-inokko/experimental/LTX-2
ROOT=/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/structural_trust_map/two_scale_static
RUN_DIR="$ROOT/stage2_dual_partition_seed43_hand_mask"
CONFIG="$ROOT/assets/ref_partition_config.json"
MASK=/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/seed_43_stage2_hand_drawn_mask.png
CACHE="$RUN_DIR/stage2_input_cache.safetensors"

PROMPT='Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure.'
NEGATIVE='flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts'

CMD=(
  "$REPO/.venv/bin/python3" "$REPO/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
  --distilled-checkpoint-path "$REPO/models/ltx-2.3-22b-distilled-1.1.safetensors"
  --spatial-upsampler-path "$REPO/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  --gemma-root /home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized
  --ic-lora-path "$REPO/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
  --reference-video /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/source_video.mp4
  --conditioning-video /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/source_video.mp4
  --reference-image /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates/flux/translated_first_frame.png
  --prompt "$PROMPT" --negative-prompt "$NEGATIVE"
  --seed 43 --stage-2-noise-seed 43
  --width 1280 --height 768 --num-frames 129 --frame-rate 24
  --image-strength 1.0 --image-crf 33 --video-strength 1.0
  --conditioning-mode rgb --conditioning-attention-strength 1.0
  --stage-1-ic-lora-strength 1.0 --stage-2-ic-lora-strength 1.0
  --first-frame-attention-schedule constant --first-frame-attention-multiplier 1.0
  --source-strength-schedule constant --source-strength 1.0
  --source-strength-routing target-queries-only
  --reference-downscale-factor-override 2
  --ref-partition-config "$CONFIG"
  --load-stage-2-input "$CACHE"
  --stage-2-branch-mode dual
  --stage-2-lora-execution runtime
  --stage-2-dual-image-ic-lora
  --stage-2-kv-mode partition
  --stage-2-kv-strength 1.0
  --stage-2-kv-layers all
  --stage-2-kv-cache-backend cpu
  --stage-2-routing-mask "$MASK"
  --stage-2-dress-video-contribution 0.0
  --quantization fp8-cast
  --output "$RUN_DIR/output.mp4"
)

printf '%q ' "${CMD[@]}" > "$RUN_DIR/command_from_cache.txt"
printf '\n' >> "$RUN_DIR/command_from_cache.txt"
cd "$REPO"
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/run_from_cache.log"
