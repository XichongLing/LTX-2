#!/usr/bin/env bash
set -euo pipefail

LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"
LTX_PYTHON="$LTX_ROOT/.venv/bin/python3"
RUNNER_SCRIPT="$LTX_ROOT/scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"

SOURCE_VIDEO="/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000.mp4"
SOURCE_FIRST_FRAME="/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000_frame0000.png"
EDITED_FIRST_FRAME="/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000_frame0000_edited.png"

OUTPUT_ROOT="/home/xichong-drz-inokko/experimental/Image_seq_transl/_draft_runs/dual_traj/elevator"
SNAPSHOT_DIR="$OUTPUT_ROOT/source_snapshot"
COMMON_STAGE2_INPUT="$OUTPUT_ROOT/common_stage2_input.safetensors"
COMMON_STAGE1_SNAPSHOT="$OUTPUT_ROOT/common_stage1_snapshot.mp4"

DISTILLED_CHECKPOINT_PATH="$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="$LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
IC_LORA_PATH="$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

FRAME_RATE=16
NUM_FRAMES=81
SEED=42
WIDTH=1280
# 1280x704 keeps the source aspect ratio closer, but source-video conditioning
# fails in the VAE with an odd intermediate latent height. Use the known-good
# Anna canvas so the two compared runs share one valid geometry.
HEIGHT=768
IMAGE_STRENGTH=1.0
VIDEO_STRENGTH=1.0
CONDITIONING_ATTENTION_STRENGTH=1.0
STAGE_2_CONDITIONING_ATTENTION_STRENGTH=1.0

PROMPT="Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in the appearance, color, materials, lighting, and style of the edited first frame. Preserve the original composition, shot timing, scene geometry, and object identity while using the RGB source video for structure when source-video conditioning is enabled."
NEGATIVE_PROMPT="flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"

for required_path in \
  "$LTX_PYTHON" \
  "$RUNNER_SCRIPT" \
  "$SOURCE_VIDEO" \
  "$SOURCE_FIRST_FRAME" \
  "$EDITED_FIRST_FRAME" \
  "$DISTILLED_CHECKPOINT_PATH" \
  "$SPATIAL_UPSAMPLER_PATH" \
  "$GEMMA_ROOT" \
  "$IC_LORA_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'Missing required path: %s\n' "$required_path" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT" "$SNAPSHOT_DIR"
cp -p "$SOURCE_VIDEO" "$SNAPSHOT_DIR/"
cp -p "$SOURCE_FIRST_FRAME" "$SNAPSHOT_DIR/"
cp -p "$EDITED_FIRST_FRAME" "$SNAPSHOT_DIR/"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,duration -of json "$SOURCE_VIDEO" > "$SNAPSHOT_DIR/source_video.ffprobe.json"
identify "$SOURCE_FIRST_FRAME" "$EDITED_FIRST_FRAME" > "$SNAPSHOT_DIR/input_images.identify.txt"
git -C "$LTX_ROOT" status --short > "$SNAPSHOT_DIR/ltx2_git_status.txt"
git -C "$LTX_ROOT" diff -- . > "$SNAPSHOT_DIR/ltx2_git_diff.patch"
cp -p "$0" "$SNAPSHOT_DIR/$(basename "$0")"

write_manifest() {
  local manifest_path="$1"
  local run_name="$2"
  local output_video="$3"
  local command_file="$4"
  "$LTX_PYTHON" - "$manifest_path" "$run_name" "$output_video" "$command_file" <<'PY'
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_name = sys.argv[2]
output_video = Path(sys.argv[3])
command_file = Path(sys.argv[4])

distilled = Path("/home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors")
upsampler = Path("/home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
adapter = Path("/home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
gemma_root = Path("/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized")
source_video = Path("/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000.mp4")
source_first = Path("/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000_frame0000.png")
edited_first = Path("/home/xichong-drz-inokko/experimental/data/dual_run/elevator/C01_elevator_robot_seed_3000_frame0000_edited.png")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha256_file(path).encode())
        h.update(b"\0")
    return h.hexdigest()

def run_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None

def meminfo() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0]) * 1024
    except Exception:
        pass
    return values

torch_info = {}
try:
    import torch

    torch_info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_bytes": torch.cuda.get_device_properties(idx).total_memory,
            }
            for idx in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
except Exception as exc:
    torch_info = {"error": repr(exc)}

output_info = {"path": str(output_video), "exists": output_video.is_file()}
if output_video.is_file():
    output_info["sha256"] = sha256_file(output_video)
    output_info["ffprobe"] = run_text(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(output_video),
        ]
    )

manifest = {
    "run_name": run_name,
    "command": command_file.read_text() if command_file.is_file() else None,
    "output": output_info,
    "assets": {
        "distilled_checkpoint": {"path": str(distilled), "sha256": sha256_file(distilled)},
        "video_audio_vae_note": "LTX video/audio VAE weights are loaded from the distilled checkpoint.",
        "video_audio_vae_sha256": sha256_file(distilled),
        "spatial_upsampler": {"path": str(upsampler), "sha256": sha256_file(upsampler)},
        "ic_lora_adapter": {"path": str(adapter), "sha256": sha256_file(adapter)},
        "gemma_root": {"path": str(gemma_root), "tree_sha256": sha256_tree(gemma_root)},
        "source_video": {"path": str(source_video), "sha256": sha256_file(source_video)},
        "source_first_frame": {"path": str(source_first), "sha256": sha256_file(source_first)},
        "edited_first_frame": {"path": str(edited_first), "sha256": sha256_file(edited_first)},
    },
    "runtime": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch_info,
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ram": meminfo(),
        "pip_freeze": run_text([sys.executable, "-m", "pip", "freeze"]),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

validate_output() {
  local run_dir="$1"
  local output_video="$2"
  local command_file="$3"
  local expected_stage2_iclora="$4"
  local expected_branch_mode="$5"
  local expect_cache_flag="$6"

  test -f "$output_video"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,duration -of json "$output_video" > "$run_dir/output.ffprobe.json"
  "$LTX_PYTHON" - "$run_dir/output.ffprobe.json" <<'PY'
import json
import sys
from fractions import Fraction
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
stream = data["streams"][0]
width = int(stream["width"])
height = int(stream["height"])
frames = int(stream["nb_frames"])
fps = Fraction(stream["r_frame_rate"])
if (width, height, frames, fps) != (1280, 768, 81, Fraction(16, 1)):
    raise SystemExit(f"Unexpected output metadata: width={width} height={height} frames={frames} fps={fps}")
PY
  grep -q -- "--conditioning-mode rgb" "$command_file"
  grep -q -- "--stage-2-ic-lora-strength $expected_stage2_iclora" "$command_file"
  if grep -q -- "--no-source-video-conditioning" "$command_file"; then
    printf 'Unexpected --no-source-video-conditioning in %s\n' "$command_file" >&2
    exit 1
  fi
  if [[ "$expected_branch_mode" != "legacy" ]]; then
    grep -q -- "--stage-2-branch-mode $expected_branch_mode" "$command_file"
  fi
  if [[ "$expect_cache_flag" == "save" ]]; then
    grep -q -- "--save-stage-2-input $COMMON_STAGE2_INPUT" "$command_file"
  elif [[ "$expect_cache_flag" == "load" ]]; then
    grep -q -- "--load-stage-2-input $COMMON_STAGE2_INPUT" "$command_file"
  fi
  sha256sum "$output_video" > "$run_dir/output.sha256"
}

run_case() {
  local run_name="$1"
  local stage2_iclora="$2"
  local branch_mode="$3"
  local cache_flag="$4"

  local run_dir="$OUTPUT_ROOT/$run_name"
  local work_dir="$run_dir/intermediates"
  local output_video="$run_dir/output.mp4"
  local command_file="$run_dir/command.txt"
  local run_log="$run_dir/run.log"

  mkdir -p "$run_dir"
  local cmd=(
    "$LTX_PYTHON" "$RUNNER_SCRIPT"
    --reference-video "$SOURCE_VIDEO"
    --conditioning-video "$SOURCE_VIDEO"
    --reference-image "$EDITED_FIRST_FRAME"
    --output "$output_video"
    --prompt "$PROMPT"
    --negative-prompt "$NEGATIVE_PROMPT"
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
    --gemma-root "$GEMMA_ROOT"
    --ic-lora-path "$IC_LORA_PATH"
    --conditioning-mode rgb
    --seed "$SEED"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num-frames "$NUM_FRAMES"
    --frame-rate "$FRAME_RATE"
    --image-strength "$IMAGE_STRENGTH"
    --image-crf 33
    --video-strength "$VIDEO_STRENGTH"
    --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
    --stage-2-ic-lora-strength "$stage2_iclora"
    --stage-2-conditioning-attention-strength "$STAGE_2_CONDITIONING_ATTENTION_STRENGTH"
    --quiet-layer-streaming
    --quantization fp8-cast
  )
  if [[ "$branch_mode" != "legacy" ]]; then
    cmd+=(--stage-2-branch-mode "$branch_mode")
  fi
  if [[ "$cache_flag" == "save" ]]; then
    cmd+=(--save-stage-2-input "$COMMON_STAGE2_INPUT")
  elif [[ "$cache_flag" == "load" ]]; then
    cmd+=(--load-stage-2-input "$COMMON_STAGE2_INPUT")
  fi

  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  write_manifest "$run_dir/environment_manifest.pre.json" "$run_name" "$output_video" "$command_file"

  printf 'Running %s -> %s\n' "$run_name" "$output_video"
  "${cmd[@]}" 2>&1 | tee "$run_log"

  validate_output "$run_dir" "$output_video" "$command_file" "$stage2_iclora" "$branch_mode" "$cache_flag"
  write_manifest "$run_dir/environment_manifest.json" "$run_name" "$output_video" "$command_file"
}

run_stage1_snapshot() {
  local run_dir="$OUTPUT_ROOT/common_stage1_snapshot"
  local work_dir="$run_dir/intermediates"
  local command_file="$run_dir/command.txt"
  local run_log="$run_dir/run.log"

  mkdir -p "$run_dir"
  local cmd=(
    "$LTX_PYTHON" "$RUNNER_SCRIPT"
    --reference-video "$SOURCE_VIDEO"
    --conditioning-video "$SOURCE_VIDEO"
    --reference-image "$EDITED_FIRST_FRAME"
    --output "$COMMON_STAGE1_SNAPSHOT"
    --prompt "$PROMPT"
    --negative-prompt "$NEGATIVE_PROMPT"
    --distilled-checkpoint-path "$DISTILLED_CHECKPOINT_PATH"
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER_PATH"
    --gemma-root "$GEMMA_ROOT"
    --ic-lora-path "$IC_LORA_PATH"
    --conditioning-mode rgb
    --seed "$SEED"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num-frames "$NUM_FRAMES"
    --frame-rate "$FRAME_RATE"
    --image-strength "$IMAGE_STRENGTH"
    --image-crf 33
    --video-strength "$VIDEO_STRENGTH"
    --conditioning-attention-strength "$CONDITIONING_ATTENTION_STRENGTH"
    --stage-2-ic-lora-strength 0.0
    --stage-2-conditioning-attention-strength "$STAGE_2_CONDITIONING_ATTENTION_STRENGTH"
    --skip-stage-2
    --quiet-layer-streaming
    --quantization fp8-cast
  )

  printf '%q ' "${cmd[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  printf 'Running common Stage-1 snapshot -> %s\n' "$COMMON_STAGE1_SNAPSHOT"
  "${cmd[@]}" 2>&1 | tee "$run_log"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,duration -of json "$COMMON_STAGE1_SNAPSHOT" > "$run_dir/output.ffprobe.json"
  sha256sum "$COMMON_STAGE1_SNAPSHOT" > "$run_dir/output.sha256"
}

run_case "stage2_image_only_no_vcond_no_iclora" "0.0" "image" "save"
run_case "stage2_image_video_iclora" "1.0" "video" "load"
run_stage1_snapshot


printf 'Elevator Stage-2 conditioning comparison completed: %s\n' "$OUTPUT_ROOT"
