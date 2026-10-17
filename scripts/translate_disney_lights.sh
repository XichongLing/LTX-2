#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

DATA_ROOT="/home/xichong-drz-inokko/experimental/data/disney"
OUTPUT_ROOT_BASE="/home/xichong-drz-inokko/experimental/Image_seq_transl"
NUM_FRAMES=161
FRAME_RATE=24

LTX_PROMPT="Follow the motion, timing, camera movement, scene structure, object layout, and action progression of the source video, but render the full sequence as a photorealistic live-action cinematic shot. Preserve the geometry, framing, poses, and spatial relationships from the source video. Use a clearly animated rotating two-light rig around the characters and foreground objects. One light is a saturated purple spotlight and the other is a warm yellow spotlight placed opposite it, 180 degrees apart. Over the 121-frame clip at 24 fps, rotate both lights clockwise around the subjects by about 180 degrees, at a steady speed of roughly 36 degrees per second. At frame 1, the purple light is frontal from the camera direction and the yellow light is behind or side-back, creating a purple-lit front with faint warm rim light. Around the middle of the clip, the lights have rotated about 90 degrees, so purple comes from one side and yellow comes from the opposite side, creating split-color side lighting. By the final frames, the yellow light has rotated to the front and the purple light has moved behind or side-back, creating yellow front illumination with violet rim light. The colored highlights and shadows must visibly travel across faces, bodies, props, and surfaces as the lights rotate. Keep the rotation smooth and continuous with no flicker, and render realistic shadows, reflections, material response, skin or surface detail, depth, and cinematic contrast."

FLUX_PROMPT="Convert this rough render into a polished photorealistic cinematic opening frame while preserving the original camera angle, composition, object positions, scene structure, and poses. Establish frame 1 of a rotating two-light rig around the characters: a saturated purple spotlight is frontal from the camera direction, strongly illuminating front-facing surfaces, while a warm yellow spotlight is behind or side-back, creating subtle golden rim light at the edges. The image should clearly look like the starting position of a lighting rotation, where the yellow light will rotate toward the front later in the shot and the purple light will move toward the back. Make the lighting physically plausible and visually obvious, with realistic shadows, material response, reflections, texture detail, and live-action movie realism."

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
