#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

INPUT_VIDEO="/home/xichong-drz-inokko/experimental/data/sintel_subset/alley_1"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/sintel_arri_min2"
INTERMEDIATE_ROOT="$OUTPUT_ROOT/intermediates"
CONTROL_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/controls"
OUT_MP4="$OUTPUT_ROOT/output.mp4"
DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/depth.mp4"

LTX_PROMPT="Follow the motion and camera of the source sequence, but render it in the look, materials, lighting, and style implied by the translated first frame, make objects and the environments look like photorealistic ones in a live-action movie."

FLUX_PROMPT="Photorealistic film still, shot on 35mm film with the look of Kodak Portra. The pose, composition, framing, and spatial content of the source image are preserved exactly. Every surface in the frame is rendered as a physical material under real light, with microsurface detail, subsurface behavior where appropriate, and accurate response to lighting. Where humans appear: skin shows pores and subsurface translucency; hair shows individual strand separation with flyaways and natural clumping; eyes show wet catchlights and iris detail."

mkdir -p "$OUTPUT_ROOT" "$CONTROL_INTERMEDIATE_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/translate_single_sequence.py"
  --dataset mpi-sintel
  --input-path "$INPUT_VIDEO"
  --output "$OUT_MP4"
  --work-dir "$INTERMEDIATE_ROOT"
  --reference-video-fps 24
  --prompt "$LTX_PROMPT"
  --use-local-flux-runner
  --flux-python "$FLUX_PYTHON"
  --flux-runner-path "$FLUX_ROOT/run_flux_first_frame_style_transfer.py"
  --flux-output-name "flux/translated_first_frame.png"
  --flux-prompt "$FLUX_PROMPT"
  --distilled-checkpoint-path /home/xichong-drz-inokko/Downloads/ltx-2-19b-distilled.safetensors
  --spatial-upsampler-path /home/xichong-drz-inokko/experimental/ComfyUI/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors
  --gemma-root /home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized
  --ic-lora-path /home/xichong-drz-inokko/experimental/ComfyUI/models/loras/LTX-2/ltx-2-19b-ic-lora-depth-control.safetensors
  --conditioning-mode depth
  --depth-output "$DEPTH_MP4"
  --width 1280
  --height 704
  --image-strength 1.0
  --video-strength 1.0
  --conditioning-attention-strength 1.0
  --quiet-layer-streaming
  --quantization fp8-cast
)

if [[ -d "$VIDEO_DEPTH_ANYTHING_ROOT" ]]; then
  COMMAND+=(
    --depth-backend video-depth-anything
    --video-depth-anything-root "$VIDEO_DEPTH_ANYTHING_ROOT"
  )
  if [[ -x "$VDA_PYTHON" ]]; then
    COMMAND+=(--video-depth-anything-python "$VDA_PYTHON")
  fi
else
  COMMAND+=(--depth-model depth-anything/Depth-Anything-V2-Small-hf)
fi

"${COMMAND[@]}"
