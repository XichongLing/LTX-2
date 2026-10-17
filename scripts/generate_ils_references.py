#!/usr/bin/env python3
"""Generate IC-LoRA reference videos with the local ILS edge-aware filter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np

ILS_SRC = Path("/home/xichong-drz-inokko/experimental/ILS/src")
if str(ILS_SRC) not in sys.path:
    sys.path.insert(0, str(ILS_SRC))

from ils_filter import lnorm_filter, welsch_filter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create edge-aware-filtered reference videos for IC-LoRA training."
    )
    parser.add_argument("dataset_json", type=Path, help="Dataset JSON with caption/media_path rows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Reference video directory. Defaults to <dataset-dir>/references_ils.",
    )
    parser.add_argument(
        "--method",
        choices=("lnorm", "welsch"),
        default="lnorm",
        help="ILS filter method.",
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None, help="ILS smoothing strength.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of ILS iterations.")
    parser.add_argument("--p", type=float, default=0.8, help="L-norm power for --method lnorm.")
    parser.add_argument("--eps", type=float, default=1e-4, help="Charbonnier epsilon for --method lnorm.")
    parser.add_argument("--gamma", type=float, default=10.0 / 255.0, help="Welsch gamma for --method welsch.")
    parser.add_argument("--gpu", action="store_true", help="Use the optional CuPy/CUDA ILS backend.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device for --gpu.")
    parser.add_argument("--override", action="store_true", help="Regenerate existing reference videos.")
    parser.add_argument("--crf", type=str, default="18", help="libx264 CRF for reference video encoding.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_json = args.dataset_json.resolve()
    dataset_dir = dataset_json.parent
    output_dir = (args.output_dir or dataset_dir / "references_ils").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with dataset_json.open("r", encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)

    filter_fn = _make_filter(args)
    updated: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        media_path = dataset_dir / sample["media_path"]
        reference_path = output_dir / f"{media_path.stem}_ils.mp4"
        rel_reference_path = reference_path.relative_to(dataset_dir)

        print(f"[{index}/{len(samples)}] {media_path.name} -> {rel_reference_path}", flush=True)
        if args.override or not reference_path.exists():
            generate_reference_video(
                media_path=media_path,
                reference_path=reference_path,
                filter_fn=filter_fn,
                crf=args.crf,
            )

        row = dict(sample)
        row["reference_path"] = str(rel_reference_path)
        updated.append(row)

    with dataset_json.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated {dataset_json} with reference_path for {len(updated)} samples.")
    return 0


def _make_filter(args: argparse.Namespace):
    if args.gpu:
        from ils_filter.gpu import lnorm_filter_gpu, welsch_filter_gpu

        if args.method == "lnorm":
            return lambda image: lnorm_filter_gpu(
                image,
                lambda_=1.0 if args.lambda_ is None else args.lambda_,
                p=args.p,
                eps=args.eps,
                iterations=4 if args.iterations is None else args.iterations,
                device=args.device,
            )
        return lambda image: welsch_filter_gpu(
            image,
            lambda_=30.0 if args.lambda_ is None else args.lambda_,
            gamma=args.gamma,
            iterations=10 if args.iterations is None else args.iterations,
            device=args.device,
        )

    if args.method == "lnorm":
        return lambda image: lnorm_filter(
            image,
            lambda_=1.0 if args.lambda_ is None else args.lambda_,
            p=args.p,
            eps=args.eps,
            iterations=4 if args.iterations is None else args.iterations,
        )
    return lambda image: welsch_filter(
        image,
        lambda_=30.0 if args.lambda_ is None else args.lambda_,
        gamma=args.gamma,
        iterations=10 if args.iterations is None else args.iterations,
    )


def generate_reference_video(
    media_path: Path,
    reference_path: Path,
    filter_fn,
    crf: str,
) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = reference_path.with_name(reference_path.stem + ".tmp" + reference_path.suffix)
    if temp_path.exists():
        temp_path.unlink()

    with av.open(str(media_path)) as input_container:
        input_stream = input_container.streams.video[0]
        fps = input_stream.average_rate or input_stream.base_rate or 24

        with av.open(str(temp_path), mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=fps)
            output_stream.width = input_stream.codec_context.width
            output_stream.height = input_stream.codec_context.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": crf}

            frame_count = 0
            for frame in input_container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                filtered = filter_fn(rgb)
                filtered = np.clip(filtered, 0.0, 1.0)
                filtered_u8 = np.rint(filtered * 255.0).astype(np.uint8)
                out_frame = av.VideoFrame.from_ndarray(filtered_u8, format="rgb24")
                for packet in output_stream.encode(out_frame):
                    output_container.mux(packet)

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  processed {frame_count} frames", flush=True)

            for packet in output_stream.encode():
                output_container.mux(packet)

    os.replace(temp_path, reference_path)


if __name__ == "__main__":
    raise SystemExit(main())
