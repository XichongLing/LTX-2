#!/usr/bin/env python3
"""Resumable causal study for the Anna dress-collapse discrepancy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = REPO_ROOT / ".venv/bin/python3"
CURRENT_RUNNER = REPO_ROOT / "scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
OUTPUT_ROOT = REPO_ROOT / "exps/stage2_collapse/anna_dress_causal_20261001"
HISTORICAL_COMMIT = "757fc2f"
SNAPSHOT_ROOT = Path("/tmp/ltx2_stage2_collapse_757fc2f")

DATA_ROOT = Path("/home/xichong-drz-inokko/experimental/Image_seq_transl/test_mp4/anna_dress/intermediates")
SOURCE_VIDEO = DATA_ROOT / "source_video.mp4"
REFERENCE_IMAGE = DATA_ROOT / "flux/translated_first_frame.png"
CHECKPOINT = REPO_ROOT / "models/ltx-2.3-22b-distilled-1.1.safetensors"
UPSCALER = REPO_ROOT / "models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
IC_LORA = REPO_ROOT / "models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
GEMMA = Path("/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized")

HISTORICAL_OUTPUT = Path(
    "/home/xichong-drz-inokko/experimental/Image_seq_transl/"
    "ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/output.mp4"
)
LOW_ROUTED_OUTPUT = REPO_ROOT / "exps/phi_noise/anna_dress_121/gaussian__source_full/output.mp4"

PROMPT = (
    "Follow the motion, timing, camera movement, and scene layout of the source video, "
    "but render the result in the appearance, color, materials, lighting, and style of "
    "the reference image. Preserve the original composition, shot timing, scene geometry, "
    "and character identity while using the RGB source video for structure."
)
NEGATIVE_PROMPT = (
    "flicker, temporal jitter, warped geometry, changed composition, changed identity, "
    "noisy grain, blurry details, compression artifacts"
)
SHORT_PROMPT = "Preserve the source motion and layout while following the edited first-frame appearance."


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    implementation: str
    branch: str = "legacy"
    lora_execution: str = "auto"
    noise_bundle_load: str | None = None
    noise_bundle_save: str | None = None
    cache_mode: str = "none"
    seed: int = 42
    width: int = 1280
    height: int = 768
    num_frames: int = 130
    prompt_variant: str = "historical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_historical_snapshot() -> Path:
    runner = SNAPSHOT_ROOT / "scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
    if runner.is_file():
        return runner
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", HISTORICAL_COMMIT],
            cwd=REPO_ROOT,
            stdout=archive,
            check=True,
        )
        archive.flush()
        with tarfile.open(archive.name) as payload:
            payload.extractall(SNAPSHOT_ROOT, filter="data")
    return runner


def common_args(
    *,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    prompt_variant: str,
    include_num_frames: bool = True,
) -> list[str]:
    prompt = PROMPT if prompt_variant == "historical" else SHORT_PROMPT
    args = [
        "--distilled-checkpoint-path",
        str(CHECKPOINT),
        "--spatial-upsampler-path",
        str(UPSCALER),
        "--gemma-root",
        str(GEMMA),
        "--ic-lora-path",
        str(IC_LORA),
        "--reference-video",
        str(SOURCE_VIDEO),
        "--conditioning-video",
        str(SOURCE_VIDEO),
        "--reference-image",
        str(REFERENCE_IMAGE),
        "--image-condition",
        str(REFERENCE_IMAGE),
        "0",
        "1.0",
        "33",
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--width",
        str(width),
        "--height",
        str(height),
        "--frame-rate",
        "24",
        "--video-strength",
        "1.0",
        "--conditioning-mode",
        "rgb",
        "--conditioning-attention-strength",
        "1.0",
        "--stage-2-ic-lora-strength",
        "1.0",
        "--stage-2-conditioning-attention-strength",
        "1.0",
        "--quantization",
        "fp8-cast",
        "--quiet-layer-streaming",
    ]
    if prompt_variant == "historical":
        args.extend(["--negative-prompt", NEGATIVE_PROMPT])
    if include_num_frames:
        args.extend(["--num-frames", str(num_frames)])
    return args


def build_command(spec: RunSpec, output_root: Path) -> list[str]:
    run_dir = output_root / spec.run_id
    output = run_dir / "output.mp4"
    if spec.implementation == "historical":
        runner = prepare_historical_snapshot()
        return [
            str(PYTHON),
            str(runner),
            *common_args(
                seed=spec.seed,
                width=spec.width,
                height=spec.height,
                num_frames=spec.num_frames,
                prompt_variant=spec.prompt_variant,
                include_num_frames=False,
            ),
            "--output",
            str(output),
        ]

    cache = output_root / "shared_full_oldcfg_stage2_input.safetensors"
    command = [
        str(PYTHON),
        str(CURRENT_RUNNER),
        *common_args(
            seed=spec.seed,
            width=spec.width,
            height=spec.height,
            num_frames=spec.num_frames,
            prompt_variant=spec.prompt_variant,
        ),
        "--output",
        str(output),
        "--stage-2-branch-mode",
        spec.branch,
        "--stage-2-lora-execution",
        spec.lora_execution,
        "--stage-2-noise-mode",
        "gaussian",
        "--stage-2-prediction-dir",
        str(run_dir / "predictions"),
        "--stage-2-noise-diagnostics",
        str(run_dir / "noise_diagnostics.json"),
    ]
    if spec.cache_mode == "save":
        command.extend(["--save-stage-2-input", str(cache)])
    elif spec.cache_mode == "save_unique":
        command.extend(["--save-stage-2-input", str(run_dir / "stage2_input.safetensors")])
    elif spec.cache_mode == "load":
        command.extend(["--load-stage-2-input", str(cache), "--load-stage-2-input-ignore-metadata"])
    if spec.noise_bundle_save is not None:
        command.extend(["--stage-2-noise-bundle-save", str(output_root / spec.noise_bundle_save)])
    if spec.noise_bundle_load is not None:
        command.extend(["--stage-2-noise-bundle-load", str(output_root / spec.noise_bundle_load)])
    return command


def gate_specs() -> list[RunSpec]:
    return [
        RunSpec("commit757_legacy_full_oldcfg", "historical"),
        RunSpec(
            "current_legacy_full_oldcfg",
            "current",
            branch="legacy",
            lora_execution="fused",
            noise_bundle_save="legacy_noise_seed42.safetensors",
            cache_mode="save",
        ),
        RunSpec(
            "current_routed_full_oldcfg",
            "current",
            branch="video",
            lora_execution="runtime",
            noise_bundle_save="split_noise_seed42.safetensors",
            cache_mode="load",
        ),
    ]


def mechanics_specs() -> list[RunSpec]:
    return [
        RunSpec(
            "legacy_fused_split_noise",
            "current",
            branch="legacy",
            lora_execution="fused",
            noise_bundle_load="split_noise_seed42.safetensors",
            cache_mode="load",
        ),
        RunSpec(
            "legacy_runtime_split_noise",
            "current",
            branch="legacy",
            lora_execution="runtime",
            noise_bundle_load="split_noise_seed42.safetensors",
            cache_mode="load",
        ),
    ]


def upstream_specs() -> list[RunSpec]:
    return [
        RunSpec(
            "upstream_low_historical_prompt",
            "current",
            branch="video",
            lora_execution="runtime",
            noise_bundle_save="upstream_low_historical_noise.safetensors",
            cache_mode="save_unique",
            width=384,
            height=256,
            num_frames=121,
            prompt_variant="historical",
        ),
        RunSpec(
            "upstream_full_short_prompt",
            "current",
            branch="video",
            lora_execution="runtime",
            noise_bundle_save="upstream_full_short_noise.safetensors",
            cache_mode="save_unique",
            width=1280,
            height=768,
            num_frames=121,
            prompt_variant="short",
        ),
        RunSpec(
            "upstream_full_historical_prompt_121",
            "current",
            branch="video",
            lora_execution="runtime",
            noise_bundle_save="upstream_full_historical_121_noise.safetensors",
            cache_mode="save_unique",
            width=1280,
            height=768,
            num_frames=121,
            prompt_variant="historical",
        ),
    ]


def confirmation_specs(factor: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in (43, 44):
        if factor == "rng":
            specs.extend(
                [
                    RunSpec(
                        f"confirm_seed{seed}_legacy_fused_legacy_noise",
                        "current",
                        branch="legacy",
                        lora_execution="fused",
                        noise_bundle_save=f"legacy_noise_seed{seed}.safetensors",
                        cache_mode="none",
                        seed=seed,
                    ),
                    RunSpec(
                        f"confirm_seed{seed}_legacy_fused_split_noise",
                        "current",
                        branch="legacy",
                        lora_execution="fused",
                        noise_bundle_save=f"split_noise_seed{seed}.safetensors",
                        cache_mode="none",
                        seed=seed,
                    ),
                ]
            )
        else:
            raise ValueError(f"unsupported confirmation factor: {factor}")
    return specs


def resolution_confirmation_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in (43, 44):
        for label, width, height in (("low", 384, 256), ("full", 1280, 768)):
            specs.append(
                RunSpec(
                    f"confirm_resolution_seed{seed}_{label}_short",
                    "current",
                    branch="video",
                    lora_execution="runtime",
                    noise_bundle_save=f"confirm_resolution_seed{seed}_{label}_noise.safetensors",
                    cache_mode="save_unique",
                    seed=seed,
                    width=width,
                    height=height,
                    num_frames=121,
                    prompt_variant="short",
                )
            )
    return specs


def is_complete(spec: RunSpec, output_root: Path) -> bool:
    output = output_root / spec.run_id / "output.mp4"
    if spec.implementation == "historical":
        return output.is_file()
    return output.is_file() and output.with_suffix(".mp4.stage2.json").is_file()


def write_manifest(output_root: Path, specs: list[RunSpec], phase: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": phase,
        "historical_commit": HISTORICAL_COMMIT,
        "existing_endpoints": {
            "historical_legacy": str(HISTORICAL_OUTPUT),
            "current_routed_low_short": str(LOW_ROUTED_OUTPUT),
        },
        "assets": {
            str(path): sha256(path)
            for path in (CHECKPOINT, UPSCALER, IC_LORA, SOURCE_VIDEO, REFERENCE_IMAGE)
        },
        "runs": [spec.__dict__ for spec in specs],
    }
    (output_root / f"{phase}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("gate", "mechanics", "upstream", "confirm-rng", "confirm-resolution"),
        default="gate",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    specs = (
        gate_specs()
        if args.phase == "gate"
        else mechanics_specs()
        if args.phase == "mechanics"
        else upstream_specs()
        if args.phase == "upstream"
        else resolution_confirmation_specs()
        if args.phase == "confirm-resolution"
        else confirmation_specs("rng")
    )
    write_manifest(output_root, specs, args.phase)
    for spec in specs:
        run_dir = output_root / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(spec, output_root)
        (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        if not args.execute:
            continue
        if is_complete(spec, output_root) and not args.force:
            print(f"[resume] {spec.run_id}")
            continue
        with (run_dir / "run.log").open("w") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
