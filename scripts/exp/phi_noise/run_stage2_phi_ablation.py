#!/usr/bin/env python3
"""Resumable Stage-2 Phi-Noise diagnostic and focused-ablation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNNER = REPO_ROOT / "scripts/style_transfer_ltx23/run_ic_lora_style_transfer.py"
MANAGED_OPTIONS = {
    "--load-stage-2-input",
    "--no-source-video-conditioning",
    "--output",
    "--save-stage-2-input",
    "--stage-2-branch-mode",
    "--stage-2-ic-lora-strength",
    "--stage-2-noise-alpha",
    "--stage-2-noise-area-fraction",
    "--stage-2-noise-diagnostics",
    "--stage-2-noise-gamma",
    "--stage-2-noise-mask-mode",
    "--stage-2-noise-mode",
    "--stage-2-prediction-dir",
    "--stage-2-start-sigma",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    noise_mode: str
    conditioning: str
    lora_strength: float
    area_fraction: float = 0.05
    gamma: float = 5.0
    mask_mode: str = "area_fraction"
    alpha: int = 3
    start_sigma: float = 0.909375

    @property
    def branch_mode(self) -> str:
        return "image" if self.conditioning == "image_only" else "video"


def diagnostic_specs() -> list[RunSpec]:
    return [
        RunSpec("gaussian__image_only", "gaussian", "image_only", 0.0),
        RunSpec("gaussian__source_no_lora", "gaussian", "source_no_lora", 0.0),
        RunSpec("gaussian__source_full", "gaussian", "source_full", 1.0),
        RunSpec("phi__image_only__p05__g5", "phi", "image_only", 0.0, gamma=5.0),
        RunSpec("phi__image_only__p05__g30", "phi", "image_only", 0.0, gamma=30.0),
        RunSpec("phi__source_weak__p05__g5", "phi", "source_weak", 0.25, gamma=5.0),
        RunSpec("phi__source_weak__p05__g30", "phi", "source_weak", 0.25, gamma=30.0),
        RunSpec(
            "raw_reference_coefficients__image_only__p05__g5",
            "raw_reference_coefficients",
            "image_only",
            0.0,
        ),
        RunSpec("gaussian__image_only__sigma08", "gaussian", "image_only", 0.0, start_sigma=0.8),
    ]


def focused_specs() -> list[RunSpec]:
    conditioning = {
        "image_only": 0.0,
        "source_no_lora": 0.0,
        "source_weak": 0.25,
        "source_full": 1.0,
    }
    specs = [RunSpec(f"gaussian__{name}", "gaussian", name, strength) for name, strength in conditioning.items()]
    for fraction in (0.02, 0.05, 0.10):
        for gamma in (5.0, 10.0, 30.0):
            p_label = f"p{round(fraction * 100):02d}"
            g_label = f"g{int(gamma)}"
            specs.extend(
                RunSpec(f"phi__{name}__{p_label}__{g_label}", "phi", name, strength, fraction, gamma)
                for name, strength in conditioning.items()
            )
            for mode in ("raw_reference_coefficients", "matched_reference_coefficients"):
                specs.append(
                    RunSpec(f"{mode}__image_only__{p_label}__{g_label}", mode, "image_only", 0.0, fraction, gamma)
                )
    specs.append(RunSpec("gaussian__image_only__sigma08", "gaussian", "image_only", 0.0, start_sigma=0.8))
    assert len(specs) == 59
    assert len({spec.run_id for spec in specs}) == 59
    return specs


def official_alpha_specs() -> list[RunSpec]:
    return [
        RunSpec(
            f"phi__image_only__alpha{alpha}__g5",
            "phi",
            "image_only",
            0.0,
            mask_mode="official_alpha",
            alpha=alpha,
        )
        for alpha in (3, 4)
    ]


def _validate_base_args(base_args: list[str]) -> None:
    conflicts = sorted(MANAGED_OPTIONS.intersection(base_args))
    if conflicts:
        raise ValueError(f"base_args contains runner-managed options: {conflicts}")


def build_command(
    spec: RunSpec,
    *,
    python: str,
    runner: Path,
    base_args: list[str],
    output_dir: Path,
    cache_path: Path,
    save_cache: bool,
) -> list[str]:
    run_dir = output_dir / spec.run_id
    command = [
        python,
        str(runner),
        *base_args,
        "--output",
        str(run_dir / "output.mp4"),
        "--stage-2-branch-mode",
        spec.branch_mode,
        "--stage-2-ic-lora-strength",
        str(spec.lora_strength),
        "--stage-2-noise-mode",
        spec.noise_mode,
        "--stage-2-noise-mask-mode",
        spec.mask_mode,
        "--stage-2-noise-area-fraction",
        str(spec.area_fraction),
        "--stage-2-noise-alpha",
        str(spec.alpha),
        "--stage-2-noise-gamma",
        str(spec.gamma),
        "--stage-2-start-sigma",
        str(spec.start_sigma),
        "--stage-2-prediction-dir",
        str(run_dir / "predictions"),
        "--stage-2-noise-diagnostics",
        str(run_dir / "noise_diagnostics.json"),
    ]
    command.extend(["--save-stage-2-input" if save_cache else "--load-stage-2-input", str(cache_path)])
    return command


def is_complete(output_dir: Path, spec: RunSpec) -> bool:
    run_dir = output_dir / spec.run_id
    output = run_dir / "output.mp4"
    required = (
        output,
        output.with_suffix(output.suffix + ".stage2.json"),
        run_dir / "noise_diagnostics.json",
        run_dir / "predictions" / "manifest.json",
    )
    return all(path.is_file() for path in required)


def _first_response(run_dir: Path) -> tuple[dict[str, Any], str] | None:
    path = run_dir / "predictions" / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    steps = manifest.get("steps", [])
    if not steps or "first_model_response" not in steps[0]:
        return None
    responses = steps[0]["first_model_response"]
    key = "routed" if "routed" in responses else next(iter(responses))
    return responses[key], key


def _cosine_against_baseline(run_dir: Path, baseline_dir: Path) -> float | None:
    run_manifest = json.loads((run_dir / "predictions" / "manifest.json").read_text())
    base_manifest = json.loads((baseline_dir / "predictions" / "manifest.json").read_text())
    run_step = run_manifest["steps"][0]
    base_step = base_manifest["steps"][0]
    run_tensors = load_file(run_dir / "predictions" / run_step["file"])
    base_tensors = load_file(baseline_dir / "predictions" / base_step["file"])
    input_key = "shared_input" if "shared_input" in run_tensors else "hybrid_input"
    x0_key = "routed_x0" if "routed_x0" in run_tensors else "image_x0"
    run_velocity = (run_tensors[input_key].float() - run_tensors[x0_key].float()) / float(run_step["sigma"])
    base_velocity = (base_tensors[input_key].float() - base_tensors[x0_key].float()) / float(base_step["sigma"])
    return float(torch.nn.functional.cosine_similarity(run_velocity.flatten(), base_velocity.flatten(), dim=0))


def write_first_step_comparisons(output_dir: Path, specs: list[RunSpec]) -> None:
    baselines = {
        (spec.conditioning, spec.start_sigma): output_dir / spec.run_id
        for spec in specs
        if spec.noise_mode == "gaussian"
    }
    for spec in specs:
        if spec.noise_mode == "gaussian":
            continue
        response = _first_response(output_dir / spec.run_id)
        baseline_dir = baselines.get((spec.conditioning, spec.start_sigma))
        baseline = _first_response(baseline_dir) if baseline_dir is not None else None
        if response is None or baseline is None:
            continue
        values, branch = response
        base_values, _ = baseline
        comparison = {
            "run_id": spec.run_id,
            "gaussian_baseline": baseline_dir.name,
            "branch": branch,
            "input_norm_ratio": values["input_norm"] / base_values["input_norm"],
            "x0_norm_ratio": values["x0_norm"] / base_values["x0_norm"],
            "velocity_norm_ratio": values["velocity_norm"] / base_values["velocity_norm"],
            "velocity_cosine_similarity": _cosine_against_baseline(output_dir / spec.run_id, baseline_dir),
            "finite": bool(values["input_finite"] and values["x0_finite"] and values["velocity_finite"]),
        }
        (output_dir / spec.run_id / "first_step_comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )


def _blind_id(run_id: str) -> str:
    return f"B-{hashlib.sha256(run_id.encode()).hexdigest()[:8].upper()}"


def write_review_rubric(output_dir: Path, specs: list[RunSpec]) -> None:
    path = output_dir / "review_rubric.csv"
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(newline="") as handle:
            existing = {row["blind_id"]: row for row in csv.DictReader(handle)}
    key = {_blind_id(spec.run_id): spec.run_id for spec in specs}
    (output_dir / "review_key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    fields = [
        "blind_id",
        "motion_layout_silhouette",
        "target_appearance",
        "source_leakage",
        "temporal_stability",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in sorted(specs, key=lambda item: _blind_id(item.run_id)):
            blind_id = _blind_id(spec.run_id)
            writer.writerow(existing.get(blind_id, {"blind_id": blind_id}))


def build_contact_sheets(output_dir: Path, specs: list[RunSpec], *, page_size: int = 12) -> None:
    completed = sorted(
        (spec for spec in specs if is_complete(output_dir, spec)),
        key=lambda item: _blind_id(item.run_id),
    )
    for page_index in range(0, len(completed), page_size):
        rows: list[tuple[RunSpec, list[Image.Image]]] = []
        for spec in completed[page_index : page_index + page_size]:
            frames = [
                frame.to_image().convert("RGB")
                for frame in av.open(output_dir / spec.run_id / "output.mp4").decode(video=0)
            ]
            if not frames:
                continue
            indices = [round((len(frames) - 1) * fraction) for fraction in (0.0, 1 / 3, 2 / 3, 1.0)]
            rows.append((spec, [frames[index] for index in indices]))
        if not rows:
            continue
        thumb_width = 320
        thumb_height = round(rows[0][1][0].height * thumb_width / rows[0][1][0].width)
        label_height = 32
        sheet = Image.new("RGB", (thumb_width * 4, (thumb_height + label_height) * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, (spec, frames) in enumerate(rows):
            y = row_index * (thumb_height + label_height)
            draw.text((4, y + 7), _blind_id(spec.run_id), fill="black")
            for column, frame in enumerate(frames):
                sheet.paste(frame.resize((thumb_width, thumb_height)), (column * thumb_width, y + label_height))
        sheet.save(output_dir / f"contact_sheet_{page_index // page_size:02d}.jpg", quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="JSON with base_args and optional output_dir/python/runner/cache_path."
    )
    parser.add_argument("--suite", choices=("diagnostic", "focused", "official-alpha"), default="diagnostic")
    parser.add_argument("--execute", action="store_true", help="Run commands; otherwise only write the suite manifest.")
    parser.add_argument("--force", action="store_true", help="Rerun completed outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text())
    base_args = [str(value) for value in config["base_args"]]
    _validate_base_args(base_args)
    output_dir = Path(config.get("output_dir", config_path.with_suffix(""))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(config.get("runner", DEFAULT_RUNNER)).expanduser().resolve()
    python = str(config.get("python", sys.executable))
    cache_path = Path(config.get("cache_path", output_dir / "stage2_input.safetensors")).expanduser().resolve()
    specs = (
        diagnostic_specs()
        if args.suite == "diagnostic"
        else focused_specs()
        if args.suite == "focused"
        else official_alpha_specs()
    )
    manifest = {
        "suite": args.suite,
        "cardinality": len(specs),
        "config": str(config_path),
        "cache_path": str(cache_path),
        "runs": [asdict(spec) for spec in specs],
    }
    (output_dir / f"{args.suite}_suite.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_review_rubric(output_dir, specs)
    if args.execute:
        for spec in specs:
            if is_complete(output_dir, spec) and not args.force:
                print(f"[resume] {spec.run_id}")  # noqa: T201
                continue
            run_dir = output_dir / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            cache_manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
            save_cache = not (cache_path.is_file() and cache_manifest_path.is_file())
            command = build_command(
                spec,
                python=python,
                runner=runner,
                base_args=base_args,
                output_dir=output_dir,
                cache_path=cache_path,
                save_cache=save_cache,
            )
            (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        write_first_step_comparisons(output_dir, specs)
        build_contact_sheets(output_dir, specs)
    else:
        print(f"Wrote {len(specs)}-run {args.suite} suite manifest to {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
