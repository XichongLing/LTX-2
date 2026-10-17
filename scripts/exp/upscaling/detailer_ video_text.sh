#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"

INPUT_VIDEO="/home/xichong-drz-inokko/UniRelight/outputs/albedo_video_examples/arrival2_light-0000.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/image_detailer"
OUT_MP4="$OUTPUT_ROOT/arrival2_light-0000_disney_video_text_detailer.mp4"

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
CONDITIONING_VIDEO=""
SKIP_STAGE_2=0

LTX_PROMPT="Transform the input video into the look of a polished Disney-like animated feature film while preserving the exact camera motion, scene composition, object layout, timing, and overall action from the source video. Render the scene with charming stylized realism, warm cinematic lighting, clean readable silhouettes, expressive handcrafted shapes, soft global illumination, rich but tasteful colors, painterly material detail, smooth animation-friendly surfaces, carefully art-directed highlights and shadows, detailed architecture and props, coherent texture across frames, and a family-friendly magical movie atmosphere. Keep the motion stable and temporally consistent, with refined edges, crisp high-frequency detail, and a final-quality animated film finish."
NEGATIVE_PROMPT="flicker, temporal jitter, unstable texture, warped geometry, broken perspective, noisy grain, blocky compression artifacts, oversharpened halos, muddy details, photorealistic horror, dark gritty realism, grotesque features, distorted hands, extra fingers, changed camera path, changed composition, changed subject identity"

mkdir -p "$OUTPUT_ROOT"

export PYTORCH_ALLOC_CONF=expandable_segments:True

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

COMMAND+=("$@")

printf 'Running video-text IC-LoRA detailer:\n'
printf '  input:  %s\n' "$INPUT_VIDEO"
printf '  output: %s\n' "$OUT_MP4"
printf '  image:  none\n'
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
