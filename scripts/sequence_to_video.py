#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import av
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_frame_pattern(dataset: str, input_dir: Path, pattern: str) -> str:
    if pattern != "auto":
        return pattern
    if dataset == "mpi-sintel":
        return "frame_*.png"
    if dataset == "synfmc":
        return f"{input_dir.name}_*.png"
    if dataset == "vkitti":
        return "*.png"
    return "*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image-sequence directory into an MP4 reference video."
    )
    parser.add_argument(
        "--dataset",
        default="generic",
        help="Dataset profile used to resolve frame loading when --glob is set to auto.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the source frame sequence.",
    )
    parser.add_argument(
        "--glob",
        default="auto",
        help="Glob pattern for frame files inside --input-dir. Use auto for dataset-specific loading.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second for the generated MP4.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF value. Lower is higher quality.",
    )
    return parser.parse_args()


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def collect_frames(input_dir: Path, pattern: str) -> list[Path]:
    frames = [
        path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    frames.sort(key=_natural_key)
    if not frames:
        raise FileNotFoundError(
            f"No frames matched pattern {pattern!r} in {input_dir}"
        )
    return frames


def crop_to_even_dimensions(array: np.ndarray) -> np.ndarray:
    height = array.shape[0] // 2 * 2
    width = array.shape[1] // 2 * 2
    return array[:height, :width]


def encode_sequence_to_mp4(frames: list[Path], output_path: Path, fps: float, crf: int) -> None:
    first_image = Image.open(frames[0]).convert("RGB")
    first_array = crop_to_even_dimensions(np.asarray(first_image, dtype=np.uint8))
    height, width = first_array.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}

        for index, frame_path in enumerate(frames):
            image = Image.open(frame_path).convert("RGB")
            if image.size != first_image.size:
                raise ValueError(
                    "All frames must share the same resolution. "
                    f"Expected {first_image.size}, got {image.size} for {frame_path}"
                )
            array = first_array if index == 0 else crop_to_even_dimensions(np.asarray(image, dtype=np.uint8))
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    frame_pattern = resolve_frame_pattern(args.dataset, input_dir, args.glob)
    frames = collect_frames(input_dir, frame_pattern)
    encode_sequence_to_mp4(frames, output_path, args.fps, args.crf)
    print(f"Encoded {len(frames)} frames from {input_dir} to {output_path}")


if __name__ == "__main__":
    main()
