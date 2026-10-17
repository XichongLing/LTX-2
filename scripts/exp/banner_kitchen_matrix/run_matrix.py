#!/usr/bin/env python3
"""Resumable Banner/Kitchen Stage-1 and Stage-2 comparison matrix.

The runner deliberately keeps all production defaults in
run_ic_lora_style_transfer.py unchanged.  It creates a normal Stage-1
snapshot, multiscale snapshots, multiscale temporal/spatial Phi snapshots,
and four Stage-2 controlled branches from the normal Stage-1 cache.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PYTHON = REPO / ".venv/bin/python3"
INFER = REPO / "scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
SAM3_PYTHON = Path("/media/xichong-drz-inokko/4715db97-9866-4ec1-9eab-1f836151d77b1/segmentation/envs/sam3/bin/python")
SAM3_TOOL = Path("/media/xichong-drz-inokko/4715db97-9866-4ec1-9eab-1f836151d77b1/segmentation/tools/segment_sam3_text_video.py")
OUT_PARENT = Path("/home/xichong-drz-inokko/experimental/Image_seq_transl/_draft_runs/dual_traj")
PROMPT = (
    "Follow the motion, timing, camera movement, and scene layout of the source video, but render the result in "
    "the appearance, color, materials, lighting, and style of the reference image. Preserve the original "
    "composition, shot timing, scene geometry, and character identity while using the RGB source video for structure."
)
NEGATIVE = "flicker, temporal jitter, warped geometry, changed composition, changed identity, noisy grain, blurry details, compression artifacts"


CASES = {
    "banner": {
        "source": Path("/home/xichong-drz-inokko/experimental/data/dual_run/banner/C02_banner_drop_seed_3001.mp4"),
        "image": Path("/home/xichong-drz-inokko/experimental/data/dual_run/banner/banner_reference_1.png"),
        "concept": "woman",
        "root": OUT_PARENT / "banner_reference1_stage1_stage2_matrix_woman",
    },
    "kitchen": {
        "source": Path("/home/xichong-drz-inokko/experimental/data/dual_run/kitchen/S02_chef_cat_seed_3000.mp4"),
        "image": Path("/home/xichong-drz-inokko/experimental/data/dual_run/kitchen/kitchen_reference_1.png"),
        "concept": "chief",
        "root": OUT_PARENT / "kitchen_reference1_stage1_stage2_matrix_chief",
    },
}


def command_text(command: list[str]) -> str:
    return shlex.join(command) + "\n"


def complete(output: Path) -> bool:
    return output.is_file() and output.stat().st_size > 0 and (output.parent / "run.log").is_file()


def run(label: str, command: list[str], output: Path) -> None:
    if complete(output):
        print(f"[reuse] {label}: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "command.txt").write_text(command_text(command))
    with (output.parent / "run.log").open("w") as log:
        log.write(command_text(command))
        log.flush()
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {output.parent / 'run.log'}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{label} did not create {output}")


def run_aux(label: str, command: list[str], log_path: Path, expected: Path) -> None:
    if expected.exists():
        print(f"[reuse] {label}: {expected}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(command_text(command))
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {log_path}")


def common(case: dict[str, object]) -> list[str]:
    source = str(case["source"])
    return [
        str(PYTHON), str(INFER),
        "--distilled-checkpoint-path", str(REPO / "models/ltx-2.3-22b-distilled-1.1.safetensors"),
        "--spatial-upsampler-path", str(REPO / "models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        "--gemma-root", "/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized",
        "--ic-lora-path", str(REPO / "models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
        "--reference-video", source, "--conditioning-video", source,
        "--reference-image", str(case["image"]),
        "--prompt", PROMPT, "--negative-prompt", NEGATIVE,
        "--seed", "42", "--stage-2-noise-seed", "42",
        "--width", "1280", "--height", "768", "--num-frames", "81", "--frame-rate", "16",
        "--image-strength", "1.0", "--image-crf", "33", "--video-strength", "1.0",
        "--conditioning-mode", "rgb", "--conditioning-attention-strength", "1.0",
        "--stage-2-conditioning-attention-strength", "1.0",
        "--stage-1-ic-lora-strength", "1.0", "--stage-2-ic-lora-strength", "1.0",
        "--first-frame-attention-schedule", "constant", "--first-frame-attention-multiplier", "1.0",
        "--source-strength-schedule", "constant", "--source-strength", "1.0",
        "--source-strength-routing", "target-queries-only",
        "--reference-downscale-factor-override", "2", "--quantization", "fp8-cast",
    ]


def ensure_masks_and_partition(case: dict[str, object]) -> tuple[Path, Path]:
    root = case["root"]
    assert isinstance(root, Path)
    concept = str(case["concept"])
    source_masks = root / "sam3" / f"stage1_{concept}" / "masks"
    run_aux(
        "SAM3 source mask",
        [str(SAM3_PYTHON), str(SAM3_TOOL), "--video", str(case["source"]), "--text-prompt", concept,
         "--output-dir", str(source_masks.parent), "--all-instances", "--fps", "source", "--overlay-alpha", "0.45", "--overwrite"],
        source_masks.parent / "sam3.log", source_masks,
    )
    partition = root / "partition" / "ref_partition_config.json"
    run_aux(
        "build Stage-1 multiscale partition",
        [str(PYTHON), str(REPO / "scripts/exp/two_scale/build_reference_partition.py"),
         "--mask", str(source_masks), "--output-dir", str(partition.parent), "--num-frames", "81",
         "--stage1-width", "640", "--stage1-height", "384", "--fps", "16",
         "--inside-scale", "4", "--outside-scale", "2", "--threshold", "0.5",
         "--inside-strength", "1.0", "--outside-strength", "1.0"],
        partition.parent / "build_partition.log", partition,
    )
    return source_masks, partition


def ensure_stage2_masks(case: dict[str, object], stage1_video: Path) -> Path:
    root = case["root"]
    assert isinstance(root, Path)
    concept = str(case["concept"])
    masks = root / "sam3" / f"stage2_{concept}" / "masks"
    run_aux(
        "SAM3 Stage-2 routing mask",
        [str(SAM3_PYTHON), str(SAM3_TOOL), "--video", str(stage1_video), "--text-prompt", concept,
         "--output-dir", str(masks.parent), "--all-instances", "--fps", "source", "--overlay-alpha", "0.45", "--overwrite"],
        masks.parent / "sam3.log", masks,
    )
    return masks


def run_case(case_name: str) -> None:
    case = CASES[case_name]
    root = case["root"]
    assert isinstance(root, Path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.json").write_text(json.dumps({k: str(v) for k, v in case.items()}, indent=2) + "\n")
    base = common(case)

    normal_stage1 = root / "stage1_normal" / "output.mp4"
    run("normal Stage-1", base + ["--stage-2-ic-lora-strength", "0", "--skip-stage-2", "--output", str(normal_stage1)], normal_stage1)

    _, partition = ensure_masks_and_partition(case)
    multiscale_stage1 = root / "stage1_multiscale" / "output.mp4"
    run("multiscale Stage-1", base + ["--ref-partition-config", str(partition), "--stage-2-ic-lora-strength", "0", "--skip-stage-2", "--output", str(multiscale_stage1)], multiscale_stage1)

    temporal = root / "stage1_multiscale_phi_temporal_source_k2_g4" / "output.mp4"
    run("multiscale temporal Phi Stage-1", base + ["--ref-partition-config", str(partition), "--stage-2-ic-lora-strength", "0", "--skip-stage-2", "--phi-noise-temporal", "--phi-transform", "temporal", "--phi-k", "2", "--phi-gamma", "4", "--phi-energy-scope", "global", "--phi-phase-source", "source", "--phi-noise-log", str(temporal.parent / "phi_noise_log.json"), "--output", str(temporal)], temporal)

    spatial = root / "stage1_multiscale_phi_spatial_source_p005_g5" / "output.mp4"
    run("multiscale spatial Phi Stage-1", base + ["--ref-partition-config", str(partition), "--stage-2-ic-lora-strength", "0", "--skip-stage-2", "--phi-noise-temporal", "--phi-transform", "spatial", "--phi-gamma", "5", "--phi-energy-scope", "global", "--phi-phase-source", "source", "--phi-mask-mode", "area_fraction", "--phi-area-fraction", "0.05", "--phi-noise-log", str(spatial.parent / "phi_noise_log.json"), "--output", str(spatial)], spatial)

    stage2_masks = ensure_stage2_masks(case, normal_stage1)
    cache = root / "stage2_cache" / "stage2_input_cache.safetensors"
    image = root / "runs" / "image_branch_iclora" / "output.mp4"
    # Controlled image branch uses the image prediction only; the source-video branch is not evaluated.
    run("Stage-2 image branch", base + ["--save-stage-2-input", str(cache), "--stage-2-branch-mode", "image", "--stage-2-lora-execution", "runtime", "--output", str(image)], image)

    shared = ["--load-stage-2-input", str(cache), "--load-stage-2-input-ignore-metadata", "--stage-2-lora-execution", "runtime"]
    video = root / "runs" / "video_branch_iclora" / "output.mp4"
    run("Stage-2 image plus source video", base + shared + ["--stage-2-branch-mode", "video", "--output", str(video)], video)

    spatial_out = root / "runs" / "spatial_shared_state_woman_mask" / "output.mp4"
    run("Stage-2 shared-state spatial routing", base + shared + ["--stage-2-branch-mode", "spatial", "--stage-2-routing-mask", str(stage2_masks), "--stage-2-dress-video-contribution", "0.0", "--output", str(spatial_out)], spatial_out)

    dual = root / "runs" / "dual_kv_partition" / "output.mp4"
    run("Stage-2 dual KV partition", base + shared + ["--stage-2-branch-mode", "dual", "--stage-2-routing-mask", str(stage2_masks), "--stage-2-dress-video-contribution", "0.0", "--stage-2-dual-image-ic-lora", "--stage-2-kv-mode", "partition", "--stage-2-kv-strength", "1.0", "--stage-2-kv-layers", "all", "--stage-2-kv-cache-backend", "cpu", "--output", str(dual)], dual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("banner", "kitchen", "all"), default="all")
    args = parser.parse_args()
    for name in (CASES if args.case == "all" else (args.case,)):
        run_case(name)


if __name__ == "__main__":
    main()
