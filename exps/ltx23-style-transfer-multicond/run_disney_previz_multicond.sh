#!/usr/bin/env bash
set -euo pipefail

# Sweep all multi-conditioning modes on the Disney previz source video.
#
# Uses a pre-styled reference image (disney/images/display_0000.png) so no
# FLUX run is needed — the reference is fixed across all mode comparisons,
# making conditioning the only variable.
#
# Output layout:
#   $OUTPUT_ROOT/
#     rgb/output.mp4
#     rgb_depth/output.mp4
#     edge_rgb/output.mp4
#     edge_depth_rgb/output.mp4
#     _work/   (shared intermediates: depth.mp4, edge.mp4 cached here)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
TRANSLATE_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/translate_single_sequence.py"

INPUT_VIDEO="/home/xichong-drz-inokko/experimental/data/disney/source_frames/source_video.mp4"
REFERENCE_IMAGE="/home/xichong-drz-inokko/experimental/data/disney/images/display_0000.png"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-multicond/disney_previz"
WORK_DIR="$OUTPUT_ROOT/_work"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

VDA_ROOT="/home/xichong-drz-inokko/experimental/Video-Depth-Anything"
VDA_PYTHON="/home/xichong-drz-inokko/yes/envs/vda311/bin/python"

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
FRAME_RATE=24
REFERENCE_VIDEO_CRF=18
SEED=42
IMAGE_STRENGTH=1.0
IMAGE_CRF=33
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
QUANTIZATION="fp8-cast"

WIDTH="1280"
HEIGHT="768"
NUM_FRAMES=""
SKIP_STAGE_2=0

EDGE_LOW_THRESHOLD=100
EDGE_HIGH_THRESHOLD=200

PROMPT="Follow the motion, timing, camera movement, and scene layout of the previz source video, but render the result in the cinematic appearance, materials, lighting, and style implied by the reference image. Preserve the original composition, shot timing, scene geometry, and character identity."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT" "$WORK_DIR"

# ---------------------------------------------------------------------------
# Mode table: "mode  slug  rgb_str  depth_str  edge_str"
# ---------------------------------------------------------------------------
MODES=(
  "rgb            rgb             1.0    ""     """
  "rgb+depth      rgb_depth       1.0    0.5    """
  "edge+rgb       edge_rgb        1.0    ""     0.5"
  "edge+depth+rgb edge_depth_rgb  0.7    0.4    0.4"
)

for mode_row in "${MODES[@]}"; do
  read -r mode slug rgb_str depth_str edge_str <<< "$mode_row"

  out_dir="$OUTPUT_ROOT/$slug"
  mkdir -p "$out_dir"
  output_mp4="$out_dir/output.mp4"

  cmd=(
    "$LTX_PYTHON" "$TRANSLATE_SCRIPT"
    --dataset              gtacrime
    --input-path           "$INPUT_VIDEO"
    --translated-first-frame "$REFERENCE_IMAGE"
    --work-dir             "$WORK_DIR"
    --output               "$output_mp4"
    --reference-video-fps  "$FRAME_RATE"
    --reference-video-crf  "$REFERENCE_VIDEO_CRF"
    --prompt               "$PROMPT"
    --negative-prompt      "$NEGATIVE_PROMPT"
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
    --spatial-upsampler-path    "$SPATIAL_UPSAMPLER_PATH"
    --gemma-root               "$GEMMA_ROOT"
    --ic-lora-path             "$IC_LORA_PATH"
    --conditioning-mode        "$mode"
    --seed                     "$SEED"
    --frame-rate               "$FRAME_RATE"
    --image-strength           "$IMAGE_STRENGTH"
    --image-crf                "$IMAGE_CRF"
    --video-strength           "$VIDEO_STRENGTH"
    --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
    --edge-low-threshold       "$EDGE_LOW_THRESHOLD"
    --edge-high-threshold      "$EDGE_HIGH_THRESHOLD"
    --quiet-layer-streaming
    --quantization             "$QUANTIZATION"
  )

  [[ -n "$WIDTH"      ]] && cmd+=(--width      "$WIDTH")
  [[ -n "$HEIGHT"     ]] && cmd+=(--height     "$HEIGHT")
  [[ -n "$NUM_FRAMES" ]] && cmd+=(--num-frames "$NUM_FRAMES")
  [[ "$SKIP_STAGE_2" == "1" ]] && cmd+=(--skip-stage-2)

  [[ -n "$rgb_str"   ]] && cmd+=(--rgb-strength   "$rgb_str")
  [[ -n "$depth_str" ]] && cmd+=(--depth-strength "$depth_str")
  [[ -n "$edge_str"  ]] && cmd+=(--edge-strength  "$edge_str")

  if [[ "$mode" == *depth* ]]; then
    cmd+=(
      --depth-backend               video-depth-anything
      --video-depth-anything-root   "$VDA_ROOT"
      --video-depth-anything-python "$VDA_PYTHON"
    )
  fi

  printf '[%s] disney_previz -> %s\n' "$mode" "$output_mp4"
  printf '%q ' "${cmd[@]}" > "$out_dir/command.txt"
  printf '\n'  >> "$out_dir/command.txt"
  "${cmd[@]}" 2>&1 | tee "$out_dir/run.log"
done
