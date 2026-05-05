#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from style_transfer_preprocess import VIDEO_SUFFIXES, natural_key


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SINGLE_SEQUENCE_SCRIPT = REPO_ROOT / "scripts" / "translate_single_sequence.py"


def input_mode_for_dataset(dataset: str) -> str:
    if dataset == "gtacrime":
        return "video"
    if dataset in {"synfmc", "mpi-sintel", "vkitti"}:
        return "sequence_dir"
    return "auto"


def iter_inputs(data_root: Path, dataset: str) -> list[Path]:
    mode = input_mode_for_dataset(dataset)
    if mode == "video":
        return sorted(
            [path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES],
            key=natural_key,
        )
    if mode == "sequence_dir":
        return sorted([path for path in data_root.iterdir() if path.is_dir()], key=natural_key)

    videos = [path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    if videos:
        return sorted(videos, key=natural_key)
    return sorted([path for path in data_root.iterdir() if path.is_dir()], key=natural_key)


def output_name_for_input(input_path: Path, dataset: str, attr: str) -> str:
    if input_mode_for_dataset(dataset) == "video" or input_path.is_file():
        class_name = input_path.parent.name
        sample_name = input_path.stem
        return f"{class_name}_{sample_name}_{attr}"
    return f"{input_path.name}_{attr}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, env=env)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Batch translate all source videos/sequences under a data root. "
            "Unknown arguments are forwarded to translate-single_sequence.py."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--attr", default="style")
    parser.add_argument("--ltx-python", default=sys.executable)
    parser.add_argument("--output-name", default="output.mp4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-gt-depths", action="store_true")
    parser.add_argument("--gt-depth-dir", default=None)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() / data_root.name
    inputs = iter_inputs(data_root, args.dataset)

    selected = inputs[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        raise FileNotFoundError(f"No inputs found under {data_root} for dataset={args.dataset}")

    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    for input_path in selected:
        output_name = output_name_for_input(input_path, args.dataset, args.attr)
        run_root = output_root / output_name
        work_dir = run_root / "intermediates"
        output_path = run_root / args.output_name

        if args.skip_existing and output_path.is_file():
            print(f"Skipping existing output: {output_path}")
            continue

        command = [
            args.ltx_python,
            str(TRANSLATE_SINGLE_SEQUENCE_SCRIPT),
            "--dataset",
            args.dataset,
            "--input-path",
            str(input_path),
            "--output",
            str(output_path),
            "--work-dir",
            str(work_dir),
        ]
        if args.accept_gt_depths:
            command.append("--accept-gt-depths")
            if args.gt_depth_dir:
                command.extend(["--gt-depth-dir", args.gt_depth_dir])
            command.extend(["--gt-depth-source-root", str(data_root)])
        command.extend(passthrough)

        print(f"=== Processing {input_path.name} ===")
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in command))
            continue

        run_command(command, env)


if __name__ == "__main__":
    main()
