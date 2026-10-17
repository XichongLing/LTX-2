#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

INPUT_VIDEO="/home/xichong-drz-inokko/experimental/data/GTA_subset/0007_view2_704.mp4"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/0007_view2_704_multiframe"
INTERMEDIATE_ROOT="$OUTPUT_ROOT/intermediates"
CONTROL_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/controls"
OUT_MP4="$OUTPUT_ROOT/output.mp4"
DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/depth.mp4"
REFERENCE_FRAME_LIST=(0 90 180 270 360)
CHUNK_MODE=true

LTX_PROMPT="A photorealistic modeled and rendered video of the same scene. Preserve the reference video's camera motion, timing, object positions, silhouettes, scale relationships, scene geometry, and depth structure exactly. Convert the visual style into a high-end physically based render with realistic materials, detailed geometry, natural global illumination, accurate reflections, contact shadows, ambient occlusion, stable exposure, and filmic color response. No layout changes, no new objects, no removed objects, no shape deformation, no identity changes."

FLUX_PROMPT="Transform the source image into a photorealistic modeled and rendered scene while preserving the exact spatial structure. Keep every object, person, vehicle, road edge, building, silhouette, camera viewpoint, perspective, scale, and composition in the same place as the source. Do not add, remove, move, resize, or reshape anything. Render the scene as a high-quality physically based 3D render with realistic surface materials, fine geometric detail, natural global illumination, soft contact shadows, ambient occlusion, accurate reflections, believable texture detail, stable exposure, and filmic photoreal color grading."

mkdir -p "$OUTPUT_ROOT" "$CONTROL_INTERMEDIATE_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$LTX_PYTHON" "$LTX_ROOT/scripts/translate_single_sequence.py"
  --dataset gtacrime
  --input-path "$INPUT_VIDEO"
  --output "$OUT_MP4"
  --work-dir "$INTERMEDIATE_ROOT"
  --reference-video-fps 24
  --reference-frame-list "${REFERENCE_FRAME_LIST[@]}"
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

if [[ "$CHUNK_MODE" == "true" ]]; then
  COMMAND[1]="$LTX_ROOT/scripts/translate_single_sequence_chunked.py"
fi

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
