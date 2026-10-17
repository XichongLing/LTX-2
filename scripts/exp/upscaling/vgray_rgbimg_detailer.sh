#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"

SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/fountain_real/intermediates/source_video.mp4"
INPUT_VIDEO="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/fountain_real/output.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/upscaling/experiments"
GRAYSCALE_VIDEO="$OUTPUT_ROOT/fountain_real_source_video_grayscale.mp4"
OUT_MP4="$OUTPUT_ROOT/fountain_real_grayscale_ref_detailer.mp4"

DISTILLED_CHECKPOINT_PATH="/home/xichong-drz-inokko/Downloads/ltx-2-19b-distilled.safetensors"
SPATIAL_UPSAMPLER_PATH="/home/xichong-drz-inokko/experimental/ComfyUI/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
DETAILER_LORA_PATH="/home/xichong-drz-inokko/experimental/ComfyUI/models/loras/LTX-2/ltx-2-19b-ic-lora-detailer.safetensors"

WIDTH=""
HEIGHT=""
MAX_LONG_SIDE=""
NUM_FRAMES=""
FRAME_RATE=""
SEED=42

DETAILER_LORA_STRENGTH=1.0
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
CONDITIONING_VIDEO="$GRAYSCALE_VIDEO"
SKIP_STAGE_2=0

IMAGE_CONDITION_PATH="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/fountain_real/intermediates/flux/translated_first_frame.png"
IMAGE_CONDITION_FRAME_IDX=0
IMAGE_CONDITION_STRENGTH=0.7
IMAGE_CONDITION_CRF=33

LTX_PROMPT="Refine the input video while preserving the same composition, camera motion, timing, subject identity, and scene layout. Add crisp high-frequency detail, clean edges, coherent texture, cinematic lighting and tonemapping, stable temporally consistent surfaces, realistic material subsurface detail, make each object look like a photorealistic one in marvel film."
NEGATIVE_PROMPT="flicker, temporal jitter, warped face, distorted hands, extra fingers, mushy texture, oversharpened halos, plastic skin, noisy grain, blocky compression artifacts, changed identity, changed composition"

mkdir -p "$OUTPUT_ROOT"

export PYTORCH_ALLOC_CONF=expandable_segments:True

ffmpeg \
  -y \
  -v error \
  -i "$SOURCE_VIDEO" \
  -map 0:v:0 \
  -an \
  -vf "hue=s=0,format=yuv420p" \
  -c:v libx264 \
  -crf 18 \
  -preset veryfast \
  "$GRAYSCALE_VIDEO"

if [[ -n "$MAX_LONG_SIDE" && -z "$WIDTH" && -z "$HEIGHT" ]]; then
  if ! command -v ffprobe >/dev/null 2>&1; then
    echo "MAX_LONG_SIDE requires ffprobe, but ffprobe was not found." >&2
    exit 1
  fi

  INPUT_WIDTH="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT_VIDEO")"
  INPUT_HEIGHT="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT_VIDEO")"

  if (( INPUT_WIDTH >= INPUT_HEIGHT )); then
    WIDTH="$MAX_LONG_SIDE"
    HEIGHT=$(( INPUT_HEIGHT * MAX_LONG_SIDE / INPUT_WIDTH ))
  else
    HEIGHT="$MAX_LONG_SIDE"
    WIDTH=$(( INPUT_WIDTH * MAX_LONG_SIDE / INPUT_HEIGHT ))
  fi

  WIDTH=$(( WIDTH / 64 * 64 ))
  HEIGHT=$(( HEIGHT / 64 * 64 ))

  if (( WIDTH <= 0 || HEIGHT <= 0 )); then
    echo "Computed invalid output size ${WIDTH}x${HEIGHT} from MAX_LONG_SIDE=$MAX_LONG_SIDE." >&2
    exit 1
  fi
fi

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/detailer/run_ic_lora_detailer.py"
  --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
  --gemma-root "$GEMMA_ROOT"
  --detailer-lora-path "$DETAILER_LORA_PATH"
  --input-video "$INPUT_VIDEO"
  --output "$OUT_MP4"
  --prompt "$LTX_PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --seed "$SEED"
  --detailer-lora-strength "$DETAILER_LORA_STRENGTH"
  --video-strength "$VIDEO_STRENGTH"
  --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
  --quiet-layer-streaming
  --quantization fp8-cast
)

if [[ "$SKIP_STAGE_2" == "1" ]]; then
  COMMAND+=(--skip-stage-2)
fi

if [[ -n "$WIDTH" ]]; then
  COMMAND+=(--width "$WIDTH")
fi

if [[ -n "$HEIGHT" ]]; then
  COMMAND+=(--height "$HEIGHT")
fi

if [[ -n "$NUM_FRAMES" ]]; then
  COMMAND+=(--num-frames "$NUM_FRAMES")
fi

if [[ -n "$FRAME_RATE" ]]; then
  COMMAND+=(--frame-rate "$FRAME_RATE")
fi

if [[ -n "$CONDITIONING_VIDEO" ]]; then
  COMMAND+=(--conditioning-video "$CONDITIONING_VIDEO")
fi

if [[ -n "$IMAGE_CONDITION_PATH" ]]; then
  COMMAND+=(
    --image-condition
    "$IMAGE_CONDITION_PATH"
    "$IMAGE_CONDITION_FRAME_IDX"
    "$IMAGE_CONDITION_STRENGTH"
    "$IMAGE_CONDITION_CRF"
  )
fi

COMMAND+=("$@")

printf 'Running grayscale-reference IC-LoRA detailer:\n'
printf '  source: %s\n' "$SOURCE_VIDEO"
printf '  gray:   %s\n' "$GRAYSCALE_VIDEO"
printf '  input:  %s\n' "$INPUT_VIDEO"
printf '  video:  %s\n' "$CONDITIONING_VIDEO"
printf '  output: %s\n' "$OUT_MP4"
if [[ -n "$IMAGE_CONDITION_PATH" ]]; then
  printf '  image:  %s @ frame %s strength=%s crf=%s\n' "$IMAGE_CONDITION_PATH" "$IMAGE_CONDITION_FRAME_IDX" "$IMAGE_CONDITION_STRENGTH" "$IMAGE_CONDITION_CRF"
else
  printf '  image:  none\n'
fi
DISPLAY_WIDTH="$WIDTH"
DISPLAY_HEIGHT="$HEIGHT"
DISPLAY_NUM_FRAMES="$NUM_FRAMES"
DISPLAY_FRAME_RATE="$FRAME_RATE"
if [[ -z "$DISPLAY_WIDTH" ]]; then
  DISPLAY_WIDTH="input"
fi
if [[ -z "$DISPLAY_HEIGHT" ]]; then
  DISPLAY_HEIGHT="input"
fi
if [[ -z "$DISPLAY_NUM_FRAMES" ]]; then
  DISPLAY_NUM_FRAMES="input"
fi
if [[ -z "$DISPLAY_FRAME_RATE" ]]; then
  DISPLAY_FRAME_RATE="input"
fi
printf '  size:   %sx%s, frames=%s, fps=%s\n' "$DISPLAY_WIDTH" "$DISPLAY_HEIGHT" "$DISPLAY_NUM_FRAMES" "$DISPLAY_FRAME_RATE"

"${COMMAND[@]}"
