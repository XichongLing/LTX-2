#!/usr/bin/env bash
set -euo pipefail

# Editable Stage-2 comparison launcher.
# Change the variables in this first block for a new source video / edited frame.

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
RUNNER_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_heel_run_robot_seed_2000.mp4"
EDITED_FIRST_FRAME="/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000_frame0000_edited_6.png"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/_draft_runs/dual_traj/elevator_heel_6"

WIDTH=1280
HEIGHT=768
NUM_FRAMES=81
FRAME_RATE=16
SEED=42

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

PROMPT="Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the edited first frame. Preserve the original composition, shot timing, scene geometry, and object identity while using the RGB source video for structure when source-video conditioning is enabled."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"

IMAGE_STRENGTH=1.0
IMAGE_CRF=33
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
STAGE_2_CONDITIONING_ATTENTION_STRENGTH=1.0

COMMON_STAGE2_INPUT="$OUTPUT_ROOT/common_stage2_input.safetensors"
COMMON_STAGE1_SNAPSHOT="$OUTPUT_ROOT/common_stage1_snapshot.mp4"

export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_ROOT"

common_args=(
  --reference-video "$SOURCE_VIDEO"
  --conditioning-video "$SOURCE_VIDEO"
  --reference-image "$EDITED_FIRST_FRAME"
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
  --gemma-root "$GEMMA_ROOT"
  --ic-lora-path "$IC_LORA_PATH"
  --conditioning-mode rgb
  --seed "$SEED"
  --width "$WIDTH"
  --height "$HEIGHT"
  --num-frames "$NUM_FRAMES"
  --frame-rate "$FRAME_RATE"
  --image-strength "$IMAGE_STRENGTH"
  --image-crf "$IMAGE_CRF"
  --video-strength "$VIDEO_STRENGTH"
  --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
  --stage-2-conditioning-attention-strength "$STAGE_2_CONDITIONING_ATTENTION_STRENGTH"
  --quiet-layer-streaming
  --quantization fp8-cast
)

run_case() {
  local name="$1"
  local branch_mode="$2"
  local stage2_iclora="$3"
  local cache_flag="$4"

  local run_dir="$OUTPUT_ROOT/$name"
  local output="$run_dir/output.mp4"
  local command_file="$run_dir/command.txt"
  local log_file="$run_dir/run.log"
  mkdir -p "$run_dir"

  local cmd=(
    "$LTX_PYTHON" "$RUNNER_SCRIPT"
    "${common_args[@]}"
    --output "$output"
    --stage-2-branch-mode "$branch_mode"
    --stage-2-ic-lora-strength "$stage2_iclora"
  )

  if [[ "$cache_flag" == "save" ]]; then
    cmd+=(--save-stage-2-input "$COMMON_STAGE2_INPUT")
  elif [[ "$cache_flag" == "load" ]]; then
    cmd+=(--load-stage-2-input "$COMMON_STAGE2_INPUT")
  fi

  printf "%q " "${cmd[@]}" > "$command_file"
  printf "\n" >> "$command_file"
  printf "Running %s -> %s\n" "$name" "$output"
  "${cmd[@]}" 2>&1 | tee "$log_file"
  sha256sum "$output" > "$run_dir/output.sha256"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
    -of json "$output" > "$run_dir/output.ffprobe.json"
}

run_stage1_snapshot() {
  local run_dir="$OUTPUT_ROOT/common_stage1_snapshot"
  mkdir -p "$run_dir"
  local cmd=(
    "$LTX_PYTHON" "$RUNNER_SCRIPT"
    "${common_args[@]}"
    --output "$COMMON_STAGE1_SNAPSHOT"
    --stage-2-ic-lora-strength 0.0
    --skip-stage-2
  )
  printf "%q " "${cmd[@]}" > "$run_dir/command.txt"
  printf "\n" >> "$run_dir/command.txt"
  printf "Running common Stage-1 snapshot -> %s\n" "$COMMON_STAGE1_SNAPSHOT"
  "${cmd[@]}" 2>&1 | tee "$run_dir/run.log"
  sha256sum "$COMMON_STAGE1_SNAPSHOT" > "$run_dir/output.sha256"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
    -of json "$COMMON_STAGE1_SNAPSHOT" > "$run_dir/output.ffprobe.json"
}

# 1. Source-conditioned Stage 1, Stage-2 image-only branch, no Stage-2 IC-LoRA.
#    This also saves the common Stage-2 input cache.
run_case "stage2_image_only_no_vcond_no_iclora" "image" "0.0" "save"

# 2. Same common Stage-2 input cache, Stage-2 source-video + IC-LoRA branch.
run_case "stage2_image_video_iclora" "video" "1.0" "load"

# Optional visual decode of the common Stage-1 result.
run_stage1_snapshot

printf "Done. Outputs are under: %s\n" "$OUTPUT_ROOT"
