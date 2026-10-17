#!/usr/bin/env bash
set -euo pipefail

# Sweep all multi-conditioning modes on MPI-Sintel sequences.
#
# Modes compared:
#   rgb             — baseline, source video only
#   rgb+depth       — source video + depth map
#   edge+rgb        — source video + Canny edges
#   edge+depth+rgb  — source video + depth map + Canny edges
#
# Intermediates (depth, edge, FLUX reference image) are written to a shared
# per-sequence work directory and cached across mode runs, so depth/edge are
# computed only once per sequence even when running all four modes.
#
# Output layout:
#   $OUTPUT_ROOT/
#     rgb/            <sequence>/output.mp4
#     rgb_depth/      <sequence>/output.mp4
#     edge_rgb/       <sequence>/output.mp4
#     edge_depth_rgb/ <sequence>/output.mp4
#     _work/          <sequence>/   (shared intermediates)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
TRANSLATE_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/translate_single_sequence.py"

DATA_ROOT="/home/xichong-drz-inokko/experimental/data/sintel_subset"
OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-multicond/sintel"
WORK_ROOT="$OUTPUT_ROOT/_work"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

FLUX_PYTHON="/home/xichong-drz-inokko/experimental/.venv/bin/python3"
FLUX_RUNNER="/home/xichong-drz-inokko/experimental/flux.2_dev/run_flux_first_frame_style_transfer.py"

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
VIDEO_STRENGTH=1.0          # fallback for any condition without an explicit strength
CONDITIONING_ATTENTION_STRENGTH=1.0
QUANTIZATION="fp8-cast"

WIDTH="1280"
HEIGHT="768"
NUM_FRAMES=""
SKIP_STAGE_2=0

# Canny thresholds for edge conditioning
EDGE_LOW_THRESHOLD=100
EDGE_HIGH_THRESHOLD=200

PROMPT="Follow the motion, timing, camera movement, and scene layout of the Sintel source sequence, but render the result in the appearance, materials, lighting, and style implied by the edited reference image. Preserve the original composition, shot timing, scene geometry, and character identity."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"
FLUX_PROMPT="Photorealistic cinematic film still derived from the source frame. Preserve the exact pose, framing, camera angle, scene layout, subject identity, and spatial content. Upgrade only the rendering style: realistic materials, detailed textures, natural lighting, coherent shadows, filmic color, physically plausible surfaces, and stable high-frequency detail."

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT" "$WORK_ROOT"

# ---------------------------------------------------------------------------
# Mode table: "mode slug  rgb_str  depth_str  edge_str"
# Empty strength fields mean the flag is omitted (falls back to VIDEO_STRENGTH).
# ---------------------------------------------------------------------------
#              mode           slug               rgb    depth  edge
MODES=(
  "rgb            rgb             1.0    ""     """
  "rgb+depth      rgb_depth       1.0    0.5    """
  "edge+rgb       edge_rgb        1.0    ""     0.5"
  "edge+depth+rgb edge_depth_rgb  0.7    0.4    0.4"
)

for sequence_dir in "$DATA_ROOT"/*/; do
  [[ -d "$sequence_dir" ]] || continue
  sequence_name="$(basename "$sequence_dir")"
  work_dir="$WORK_ROOT/$sequence_name"
  mkdir -p "$work_dir"

  for mode_row in "${MODES[@]}"; do
    read -r mode slug rgb_str depth_str edge_str <<< "$mode_row"

    out_dir="$OUTPUT_ROOT/$slug/$sequence_name"
    mkdir -p "$out_dir"
    output_mp4="$out_dir/output.mp4"

    cmd=(
      "$LTX_PYTHON" "$TRANSLATE_SCRIPT"
      --dataset          mpi-sintel
      --input-path       "$sequence_dir"
      --glob             "frame_*.png"
      --work-dir         "$work_dir"
      --output           "$output_mp4"
      --reference-video-fps  "$FRAME_RATE"
      --reference-video-crf  "$REFERENCE_VIDEO_CRF"
      --prompt               "$PROMPT"
      --negative-prompt      "$NEGATIVE_PROMPT"
      --use-local-flux-runner
      --flux-python          "$FLUX_PYTHON"
      --flux-runner-path     "$FLUX_RUNNER"
      --flux-output-name     "flux/translated_first_frame.png"
      --flux-prompt          "$FLUX_PROMPT"
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

    # Per-condition strengths (omit flag when field is empty)
    [[ -n "$rgb_str"   ]] && cmd+=(--rgb-strength   "$rgb_str")
    [[ -n "$depth_str" ]] && cmd+=(--depth-strength "$depth_str")
    [[ -n "$edge_str"  ]] && cmd+=(--edge-strength  "$edge_str")

    # Video-Depth-Anything backend (only for modes that include depth)
    if [[ "$mode" == *depth* ]]; then
      cmd+=(
        --depth-backend              video-depth-anything
        --video-depth-anything-root  "$VDA_ROOT"
        --video-depth-anything-python "$VDA_PYTHON"
      )
    fi

    printf '[%s] %s -> %s\n' "$mode" "$sequence_name" "$output_mp4"
    printf '%q ' "${cmd[@]}" > "$out_dir/command.txt"
    printf '\n'  >> "$out_dir/command.txt"
    "${cmd[@]}" 2>&1 | tee "$out_dir/run.log"
  done
done
