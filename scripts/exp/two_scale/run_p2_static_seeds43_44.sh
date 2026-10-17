#!/usr/bin/env bash
set -euo pipefail

REPO=/home/xichong-drz-inokko/experimental/LTX-2
ROOT=/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/structural_trust_map/two_scale_static
CONFIG="$ROOT/assets/ref_partition_config.json"

PROMPT='Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB source video for structure.'
NEGATIVE='flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts'

for SEED in 43 44; do
  RUN_DIR="$ROOT/P2_partition_s2_s4_strength1/seed_${SEED}"
  mkdir -p "$RUN_DIR"
  cp "$CONFIG" "$RUN_DIR/resolved_ref_partition_config.json"

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
    --seed "$SEED" --width 1280 --height 768 --num-frames 129 --frame-rate 24
    --image-strength 1.0 --image-crf 33 --video-strength 1.0
    --conditioning-mode rgb --conditioning-attention-strength 1.0
    --stage-1-ic-lora-strength 1.0 --stage-2-ic-lora-strength 0.0
    --first-frame-attention-schedule constant --first-frame-attention-multiplier 1.0
    --source-strength-schedule constant --source-strength 1.0
    --source-strength-routing target-queries-only
    --source-strength-log "$RUN_DIR/source_strength_log.json"
    --reference-downscale-factor-override 2
    --ref-partition-config "$CONFIG"
    --attention-probe-output "$RUN_DIR/attention_probe.jsonl"
    --attention-probe-layers 0,8,16,24,32,40,47
    --attention-probe-steps all --attention-probe-heads mean
    --quantization fp8-cast --skip-stage-2
    --output "$RUN_DIR/output.mp4"
  )
  printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"
  printf '\n' >> "$RUN_DIR/command.txt"
  cd "$REPO"
  "${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"
done
