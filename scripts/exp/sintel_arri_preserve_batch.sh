#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

DATA_ROOT="/home/xichong-drz-inokko/experimental/data/MPI-Sintel-training_images/training/albedo"
OUTPUT_ROOT_BASE="/home/xichong-drz-inokko/experimental/Image_seq_transl/sintel_arri_preserve"
# NUM_FRAMES=161
FRAME_RATE=24

LTX_PROMPT="A realistic video with strong directional key lighting from the upper-left. All objects, people, camera motion, scene geometry, and depth structure follow the reference video exactly. Left-facing surfaces are illuminated, right-facing surfaces are darker, and cast shadows fall consistently toward the lower-right. Stable exposure, realistic contact shadows, no shape deformation, no extra objects."

FLUX_PROMPT="Photorealistic film still rendered from the source image. The pose, gaze, head position, body position, framing, composition, camera angle, and subject placement are identical to the source — nothing about the spatial content changes. Captured with the look of Kodak Portra 400 on an 85mm lens at f/2, with natural soft lighting consistent with the source image's existing lighting direction.
Only the rendering fidelity is upgraded. Skin shows fine pores, subtle redness around the nose and ears from subsurface scattering, faint asymmetry, and the soft sheen of natural oils. Hair is made of thousands of individual fibers, each strand catching light separately, with visible scalp showing through irregular partings, fine wispy flyaways at the hairline and temples, baby hairs, and a slight halo of frizz where light passes through the outer layer. Hair clumps into uneven natural sections with weight and gravity. Eyes have wet catchlights, visible iris striations, and faint vascular detail in the sclera. Clothing fabric shows weave, wrinkle, and material weight. The image has the color grading of a contemporary independent film, shallow depth of field with gentle bokeh, and the subtle organic grain of 35mm film stock. Every surface behaves like a physical material under real light, while pose, framing, and all spatial content match the source image exactly."

export PYTORCH_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/translate_sequences.py"
  --dataset mpi-sintel
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT_BASE"
  --attr lights
  --ltx-python "$LTX_PYTHON"
  # --reference-video-fps 24
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
  # --num-frames "$NUM_FRAMES"
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
