#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Configurable parameters
# -----------------------
LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
STYLE_TRANSFER_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

DATA_ROOT="/home/xichong-drz-inokko/experimental/data/disney/source_frames/source_video.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

FRAME_RATE=24
SEED=42
IMAGE_STRENGTH_first=1.0
IMAGE_STRENGTH_other=0.5
IMAGE_CRF=33
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
QUANTIZATION="fp8-cast"

WIDTH="1280"
HEIGHT="768"
NUM_FRAMES=""
SKIP_STAGE_2=0

PROMPT="Follow the motion, timing, camera movement, and scene layout of the Sintel source sequence, but render the result in the appearance, color, materials, lighting, and style implied by the reference images. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB control video for structure."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"

# Reference images and the target (pixel) frame indices they condition.
REFERENCE_IMAGE_0="/home/xichong-drz-inokko/experimental/data/disney/images/display_0000.png"
REFERENCE_IMAGE_97="/home/xichong-drz-inokko/experimental/insert-anything/examples/ac/match_without_colom.webp"

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT"

sequence_name="$(basename "$DATA_ROOT")"
sequence_output_dir="$OUTPUT_ROOT/plane_multi_ref_insertanything_0_98"
output_mp4="$sequence_output_dir/output.mp4"


mkdir -p "$sequence_output_dir"

cmd=(
"$LTX_PYTHON" "$STYLE_TRANSFER_SCRIPT"
--distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
--spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
--gemma-root "$GEMMA_ROOT"
--ic-lora-path "$IC_LORA_PATH"
--reference-video "$DATA_ROOT"
--conditioning-video "$DATA_ROOT"
--reference-image "$REFERENCE_IMAGE_0"
--image-condition "$REFERENCE_IMAGE_0" 0 "$IMAGE_STRENGTH_first" "$IMAGE_CRF"
--image-condition "$REFERENCE_IMAGE_97" 97 "$IMAGE_STRENGTH_other" "$IMAGE_CRF"
--output "$output_mp4"
--prompt "$PROMPT"
--negative-prompt "$NEGATIVE_PROMPT"
--conditioning-mode rgb
--seed "$SEED"
--frame-rate "$FRAME_RATE"
--video-strength "$VIDEO_STRENGTH"
--conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
--quiet-layer-streaming
--quantization "$QUANTIZATION"
)

[[ -n "$WIDTH" ]] && cmd+=(--width "$WIDTH")
[[ -n "$HEIGHT" ]] && cmd+=(--height "$HEIGHT")
[[ -n "$NUM_FRAMES" ]] && cmd+=(--num-frames "$NUM_FRAMES")
[[ "$SKIP_STAGE_2" == "1" ]] && cmd+=(--skip-stage-2)

printf 'Running %s -> %s\n' "$sequence_name" "$output_mp4"
printf '%q ' "${cmd[@]}" > "$sequence_output_dir/command.txt"
printf '\n' >> "$sequence_output_dir/command.txt"
"${cmd[@]}" 2>&1 | tee "$sequence_output_dir/run.log"
