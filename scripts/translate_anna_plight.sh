#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

INPUT_VIDEO="/home/xichong-drz-inokko/experimental/data/test_mp4/anna.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_plight"
INTERMEDIATE_ROOT="$OUTPUT_ROOT/intermediates"
CONTROL_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/controls"
OUT_MP4="$OUTPUT_ROOT/output.mp4"
DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/depth.mp4"

LTX_PROMPT="Follow the motion, timing, camera movement, scene structure, object layout, and action progression of the source video, but render the full sequence as a photorealistic live-action cinematic shot. Preserve the geometry, framing, poses, and spatial relationships from the source video. Apply a consistent lighting setup throughout the entire video: a single neutral white off-camera area light positioned behind the camera, illuminating the subjects and foreground from the camera direction. The light source itself must remain outside the frame and must not appear in the scene. The white camera-facing area light should create broad realistic highlights, soft physically plausible shadows, and bright neutral white illumination without colored tint. Keep this off-camera white lighting design stable across all frames as the camera and objects move. Use realistic materials, shadows, reflections, skin or surface detail, depth, and cinematic contrast."

FLUX_PROMPT="Convert this rough render into a polished photorealistic cinematic frame while preserving the original camera angle, composition, object positions, scene structure, and poses. Add a clear single neutral white off-camera area-light setup: one broad white light source positioned behind the camera, shining from the camera direction onto the subjects and foreground. The light source must stay outside the frame and must not be visible in the scene. Make the lighting physically plausible and visually obvious, with broad highlights, soft realistic shadows, natural material response, reflections, texture detail, and live-action movie realism without colored tint. The result should look like the first frame of a photorealistic film shot with a consistent off-camera white area-light design."

mkdir -p "$OUTPUT_ROOT" "$CONTROL_INTERMEDIATE_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/translate-single_sequence.py"
  --dataset gtacrime
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
