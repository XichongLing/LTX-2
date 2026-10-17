#!/bin/bash

set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
FLUX_ROOT="/home/xichong-drz-inokko/experimental/flux.2_dev"
VIDEO_DEPTH_ANYTHING_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

DATASET_NAME="gtacrime"
DATA_ROOT="/home/xichong-drz-inokko/experimental/data/GTA_subset"
ACCEPT_GT_DEPTHS=0
# GT_DEPTH_DIR="/home/xichong-drz-inokko/experimental/data/vkitti/vkitti_1.3.1_depthgt/0001"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-gt-depths)
      ACCEPT_GT_DEPTHS=1
      shift
      ;;
    --gt-depth-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --gt-depth-dir" >&2
        exit 1
      fi
      GT_DEPTH_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$ACCEPT_GT_DEPTHS" == "1" && -z "$GT_DEPTH_DIR" ]]; then
  echo "--accept-gt-depths requires --gt-depth-dir" >&2
  exit 1
fi

OUTPUT_ROOT_BASE="/home/xichong-drz-inokko/experimental/Image_seq_transl"
DATASET_FOLDER_NAME="$(basename "$DATA_ROOT")"
OUTPUT_ROOT="$OUTPUT_ROOT_BASE/$DATASET_FOLDER_NAME"


export PYTORCH_ALLOC_CONF=expandable_segments:True

if [[ "$DATASET_NAME" == "gtacrime" ]]; then
  INPUT_MODE="video"
elif [[ "$DATASET_NAME" == "synfmc" || "$DATASET_NAME" == "mpi-sintel" || "$DATASET_NAME" == "vkitti" ]]; then
  INPUT_MODE="sequence_dir"
else
  echo "Unsupported DATASET_NAME: $DATASET_NAME" >&2
  exit 1
fi

iter_inputs() {
  if [[ "$INPUT_MODE" == "video" ]]; then
    find "$DATA_ROOT" -type f -name "*.mp4" | sort
  else
    find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

while IFS= read -r INPUT_PATH; do
    ATTR="pvsimple"

    if [[ "$INPUT_MODE" == "video" ]]; then
        CLASS_NAME="$(basename "$(dirname "$INPUT_PATH")")"
        SAMPLE_NAME="$(basename "$INPUT_PATH" .mp4)"
        OUTPUT_NAME="${CLASS_NAME}_${SAMPLE_NAME}_${ATTR}"
    else
        SAMPLE_NAME="$(basename "$INPUT_PATH")"
        OUTPUT_NAME="${SAMPLE_NAME}_${ATTR}"
    fi

    RUN_ROOT="$OUTPUT_ROOT/${OUTPUT_NAME}"
    INTERMEDIATE_ROOT="$RUN_ROOT/intermediates"
    FLUX_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/flux"
    CONTROL_INTERMEDIATE_DIR="$INTERMEDIATE_ROOT/controls"
    OUT_MP4="$RUN_ROOT/output.mp4"
    if [[ "$ACCEPT_GT_DEPTHS" == "1" ]]; then
        DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/gt_depth.mp4"
    else
        DEPTH_MP4="$CONTROL_INTERMEDIATE_DIR/depth.mp4"
    fi

    mkdir -p "$RUN_ROOT" "$FLUX_INTERMEDIATE_DIR" "$CONTROL_INTERMEDIATE_DIR"

    echo "=== Processing $SAMPLE_NAME ==="

    COMMAND=(
	    "$LTX_PYTHON" "$LTX_ROOT/scripts/run_sequence_ic_lora_pipeline.py"
	    --dataset "$DATASET_NAME"
	    --input-dir "$INPUT_PATH"
	    --output "$OUT_MP4"
	    --work-dir "$INTERMEDIATE_ROOT"
	    --reference-video-fps 24
	    --prompt "Follow the motion, camera movement, timing, scene structure, and action progression of the input video, but render the full sequence in a film previsualization (previz) style consistent with the first-frame reference image. Keep the same blocking, framing, object layout, and pose evolution, while converting all assets into rough 3D proxy models with simplified geometry, placeholder materials, minimal textures, and basic neutral lighting. The output should look like an unfinished layout / previs render used for shot planning, not a polished final cinematic render."
	    --use-local-flux-runner
	    --flux-python "$FLUX_PYTHON"
	    --flux-runner-path "$FLUX_ROOT/run_flux_first_frame_style_transfer.py"
	    --flux-output-name "flux/translated_first_frame.png"
	    --flux-prompt "Edit this image into a film previsualization frame. Preserve the original camera angle, composition, object positions, scene structure, and character poses, but convert everything into a rough 3D layout render with proxy-quality meshes, simplified geometry, placeholder materials, minimal textures, basic lighting, and an unfinished production-planning look. The image should look like an early previs render from a VFX/animation pipeline, not a polished final shot, not concept art, and not a hand-drawn storyboard."
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

    if [[ "$ACCEPT_GT_DEPTHS" == "1" ]]; then
	    COMMAND+=(
		    --accept-gt-depths
		    --gt-depth-dir "$GT_DEPTH_DIR"
		    --gt-depth-source-root "$DATA_ROOT"
		    --gt-depth-source-path "$INPUT_PATH"
	    )
    elif [[ -d "$VIDEO_DEPTH_ANYTHING_ROOT" ]]; then
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
done < <(iter_inputs)
