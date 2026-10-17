#!/usr/bin/env python3
"""Run the Stage-1 global temporal Phi-noise pilot matrix.

The runner expects a baseline C1_s4 command that already contains the desired assets,
prompt, 129-frame/1280x768 settings, --reference-downscale-factor-override 4, and
--skip-stage-2. It rewrites only seed/output/log-related options and appends Phi flags.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


CONDITIONS = [
    ("B4", None, None),
    ("T1a", 1, 4.0),
    ("T1b", 1, 30.0),
    ("T2a", 2, 4.0),
    ("T2b", 2, 30.0),
    ("T3a", 3, 4.0),
    ("T3b", 3, 30.0),
]


def _replace_option(tokens: list[str], option: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(tokens):
        if tokens[i] == option:
            out.extend([option, value])
            i += 2
            replaced = True
        else:
            out.append(tokens[i])
            i += 1
    if not replaced:
        out.extend([option, value])
    return out


def _ensure_flag(tokens: list[str], flag: str) -> list[str]:
    return tokens if flag in tokens else [*tokens, flag]


def _remove_option(tokens: list[str], option: str, has_value: bool = True) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == option:
            i += 2 if has_value else 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def build_command(base_tokens: list[str], condition: str, seed: int, output_dir: Path, k: int | None, gamma: float | None) -> list[str]:
    run_dir = output_dir / f"seed_{seed}" / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    tokens = list(base_tokens)
    tokens = _replace_option(tokens, "--seed", str(seed))
    tokens = _replace_option(tokens, "--output", str(run_dir / "output.mp4"))
    tokens = _ensure_flag(tokens, "--skip-stage-2")
    tokens = _replace_option(tokens, "--reference-downscale-factor-override", "4")
    tokens = _remove_option(tokens, "--phi-noise-log")
    tokens = _remove_option(tokens, "--phi-noise-temporal", has_value=False)
    if k is not None and gamma is not None:
        tokens = _ensure_flag(tokens, "--phi-noise-temporal")
        tokens = _replace_option(tokens, "--phi-k", str(k))
        tokens = _replace_option(tokens, "--phi-gamma", str(gamma))
        tokens = _replace_option(tokens, "--phi-energy-scope", "global")
        tokens = _replace_option(tokens, "--phi-phase-source", "source")
        tokens = _replace_option(tokens, "--phi-noise-log", str(run_dir / "phi_noise_log.json"))
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-command-file", required=True, help="Text file containing the exact C1_s4 baseline command.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "phi_step1_report.md").write_text(
        "# Stage-1 Phi-noise reconnaissance report\n\n"
        "Hook point: Stage-1 video noiser in ICLoraPipeline, before denoise-mask mixing.\n"
        "Video tokens are patchified in (frame, height, width) order with patch size 1.\n"
        "Audio state is built after video state and is not transformed by Phi-noise.\n"
        "This runner uses the provided C1_s4 command as the source of truth for assets/prompts/settings.\n"
    )
    base_text = Path(args.base_command_file).expanduser().read_text().strip()
    base_tokens = shlex.split(base_text)
    manifest: dict[str, object] = {"base_command_file": str(Path(args.base_command_file).resolve()), "runs": []}

    for seed in args.seeds:
        for condition, k, gamma in CONDITIONS:
            cmd = build_command(base_tokens, condition, seed, output_root, k, gamma)
            output = Path(cmd[cmd.index("--output") + 1])
            run_dir = output.parent
            command_path = run_dir / "command.txt"
            log_path = run_dir / "run.log"
            command_path.write_text(shlex.join(cmd) + "\n")
            record = {"seed": seed, "condition": condition, "output": str(output), "command": str(command_path)}
            manifest["runs"].append(record)
            if output.exists() and not args.force:
                record["status"] = "skipped_existing"
                continue
            if args.dry_run:
                record["status"] = "dry_run"
                continue
            with log_path.open("w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            record["status"] = "ok" if proc.returncode == 0 else "failed"
            record["returncode"] = proc.returncode
            if proc.returncode != 0:
                break
    (output_root / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Stage-1 Phi-noise pilot report",
        "",
        "## Run list",
        "",
        "| Condition | Seed | Status | Output |",
        "|---|---:|---|---|",
    ]
    for run in manifest["runs"]:
        lines.append(
            f"| {run['condition']} | {run['seed']} | {run.get('status', 'unknown')} | `{run['output']}` |"
        )
    lines.extend([
        "",
        "## Blank rating sheet",
        "",
        "| Condition | Seed | Human motion vs source (closer than B4 / same / worse) | Dress holds after ~frame 17 (yes/partial/no) | Appearance vs edited frame (same as B4 / drifts toward source) | Artifacts (none / saturation / flicker / texture / other) | Notes |",
        "|---|---:|---|---|---|---|---|",
    ])
    for seed in args.seeds:
        for condition, _, _ in CONDITIONS:
            lines.append(f"| {condition} | {seed} |  |  |  |  |  |")
    (output_root / "phi_step3_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
