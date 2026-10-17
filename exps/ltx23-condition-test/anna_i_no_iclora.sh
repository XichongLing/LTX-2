#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Configurable parameters
# -----------------------
LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
TRANSLATE_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/translate_single_sequence.py"

DATA_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_dress/intermediates/source_video.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/without_iclora_no_vid"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
FLUX_RUNNER="/home/xichong-drz-inokko/experimental/flux.2_dev/run_flux_first_frame_style_transfer.py"

DATASET="gtacrime"
FRAME_RATE=24
REFERENCE_VIDEO_CRF=18
SEED=42
IMAGE_STRENGTH=1.0
IMAGE_CRF=33
VIDEO_STRENGTH=1.0
STAGE_1_IC_LORA_STRENGTH=0.0
NO_SOURCE_VIDEO_CONDITIONING=1
CONDITIONING_ATTENTION_STRENGTH=1.0
QUANTIZATION="fp8-cast"

WIDTH="1280"
HEIGHT="768"
NUM_FRAMES=""
SKIP_STAGE_2=1

PROMPT="Follow the motion, timing, camera movement, and scene layout of the Sintel source sequence, but render the result in the appearance, color, materials, lighting, and style implied by the edited reference image. Preserve the original composition, shot timing, scene geometry, and character identity while using the RGB control video for structure."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"
FLUX_PROMPT="Photorealistic cinematic film still derived from the source frame. Preserve the exact pose, framing, camera angle, scene layout, subject identity, and spatial content. Upgrade only the rendering style: realistic materials, detailed textures, natural lighting, coherent shadows, filmic color, physically plausible surfaces, and stable high-frequency detail."

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT"

sequence_dir="$DATA_ROOT"
sequence_name="$(basename "$DATA_ROOT")"
sequence_output_dir="$OUTPUT_ROOT/$sequence_name"
work_dir="$sequence_output_dir/intermediates"
output_mp4="$sequence_output_dir/output.mp4"
model_output_mp4="$output_mp4"

mkdir -p "$sequence_output_dir"

cmd=(
"$LTX_PYTHON" "$TRANSLATE_SCRIPT"
--dataset "$DATASET"
--reference-image-path /home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_dress/intermediates/flux/translated_first_frame.png
--input-path "$sequence_dir"
--glob "frame_*.png"
--work-dir "$work_dir"
--output "$model_output_mp4"
--reference-video-fps "$FRAME_RATE"
--reference-video-crf "$REFERENCE_VIDEO_CRF"
--prompt "$PROMPT"
--negative-prompt "$NEGATIVE_PROMPT"
--use-local-flux-runner
--flux-python "$FLUX_PYTHON"
--flux-runner-path "$FLUX_RUNNER"
--flux-output-name "flux/translated_first_frame.png"
--flux-prompt "$FLUX_PROMPT"
--distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
--spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
--gemma-root "$GEMMA_ROOT"
--ic-lora-path "$IC_LORA_PATH"
--stage-1-ic-lora-strength "$STAGE_1_IC_LORA_STRENGTH"
--conditioning-mode rgb
--seed "$SEED"
--frame-rate "$FRAME_RATE"
--image-strength "$IMAGE_STRENGTH"
--image-crf "$IMAGE_CRF"
--video-strength "$VIDEO_STRENGTH"
--conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
--quiet-layer-streaming
--quantization "$QUANTIZATION"
)

[[ -n "$WIDTH" ]] && cmd+=(--width "$WIDTH")
[[ -n "$HEIGHT" ]] && cmd+=(--height "$HEIGHT")
[[ -n "$NUM_FRAMES" ]] && cmd+=(--num-frames "$NUM_FRAMES")
[[ "$SKIP_STAGE_2" == "1" ]] && cmd+=(--skip-stage-2)
[[ "$NO_SOURCE_VIDEO_CONDITIONING" == "1" ]] && cmd+=(--no-source-video-conditioning)

printf 'Running %s -> %s\n' "$sequence_name" "$output_mp4"
printf '%q ' "${cmd[@]}" > "$sequence_output_dir/command.txt"
printf '\n' >> "$sequence_output_dir/command.txt"
"${cmd[@]}" 2>&1 | tee "$sequence_output_dir/run.log"

