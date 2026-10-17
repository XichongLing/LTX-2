#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

DATA_ROOT="/home/xichong-drz-inokko/experimental/data/test_mp4"
OUTPUT_ROOT_BASE="/home/xichong-drz-inokko/experimental/Image_seq_transl"
NUM_FRAMES=161
FRAME_RATE=24

LTX_PROMPT="A realistic video with strong directional key lighting from the upper-left. All objects, people, camera motion, scene geometry, and depth structure follow the reference video exactly. Left-facing surfaces are illuminated, right-facing surfaces are darker, and cast shadows fall consistently toward the lower-right. Stable exposure, realistic contact shadows, no shape deformation, no extra objects."

FLUX_PROMPT="Relight the image with a strong cinematic key light from the upper left. Preserve the exact scene structure, object positions, silhouettes, camera viewpoint, and all materials. Do not change the identity or shape of any object. Add physically plausible directional lighting only. Lighting: bright white key light from upper-left, clear highlight on left-facing surfaces, darker right-facing surfaces, visible cast shadows falling toward the lower-right, realistic contact shadows, high dynamic range, natural exposure."

export PYTORCH_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/translate_sequences.py"
  --dataset gtacrime
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT_BASE"
  --attr lights
  --ltx-python "$LTX_PYTHON"
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
  --width 1280
  --height 704
  --num-frames "$NUM_FRAMES"
  --frame-rate "$FRAME_RATE"
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
