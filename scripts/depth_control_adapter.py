#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import av
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _resolve_vkitti_depth_dir(gt_depth_root: Path, source_root: Path, source_path: Path) -> Path:
    try:
        relative_source = source_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"VKITTI source path {source_path} is not under source root {source_root}; "
            "cannot resolve matching GT-depth directory."
        ) from exc

    depth_dir = gt_depth_root / relative_source
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"VKITTI GT-depth directory not found: {depth_dir}")
    return depth_dir


def resolve_gt_depth_dir(
    dataset: str,
    gt_depth_root: str,
    source_root: str,
    source_path: str,
) -> Path:
    root = Path(gt_depth_root).expanduser().resolve()
    source_root_path = Path(source_root).expanduser().resolve()
    source_path_resolved = Path(source_path).expanduser().resolve()

    if dataset == "vkitti":
        return _resolve_vkitti_depth_dir(root, source_root_path, source_path_resolved)

    raise ValueError(f"No GT-depth adapter is implemented for dataset {dataset!r}")


def collect_depth_frames(depth_dir: Path) -> list[Path]:
    frames = sorted(
        [path for path in depth_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not frames:
        raise FileNotFoundError(f"No GT-depth image files found in {depth_dir}")
    return frames


def _load_depth_array(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32)


def _vkitti_depth_to_depth_anything_like(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid_depth = depth[valid]
    lo = np.percentile(valid_depth, 1.0)
    hi = np.percentile(valid_depth, 99.0)
    if hi <= lo:
        hi = float(valid_depth.max())
        lo = float(valid_depth.min())

    if hi > lo:
        normalized = (depth - lo) / (hi - lo)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)

    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid] = 1.0

    # VKITTI metric depth is darker for nearer pixels when viewed directly.
    # Depth Anything controls in this pipeline are brighter for nearer pixels,
    # so invert after normalization.
    control = 1.0 - normalized
    control_uint8 = np.round(control * 255.0).astype(np.uint8)
    return np.repeat(control_uint8[:, :, None], 3, axis=2)


def _crop_to_even_dimensions(frame: np.ndarray) -> np.ndarray:
    height = frame.shape[0] // 2 * 2
    width = frame.shape[1] // 2 * 2
    return frame[:height, :width]


def create_gt_depth_control_video(
    dataset: str,
    gt_depth_root: str,
    source_root: str,
    source_path: str,
    output_path: str,
    fps: float,
    frame_cap: int | None = None,
) -> Path:
    depth_dir = resolve_gt_depth_dir(dataset, gt_depth_root, source_root, source_path)
    depth_frames = collect_depth_frames(depth_dir)
    if frame_cap is not None:
        depth_frames = depth_frames[:frame_cap]
    if not depth_frames:
        raise FileNotFoundError(f"No GT-depth frames selected from {depth_dir}")

    first_depth = _crop_to_even_dimensions(_vkitti_depth_to_depth_anything_like(_load_depth_array(depth_frames[0])))
    height, width = first_depth.shape[:2]

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for index, frame_path in enumerate(depth_frames):
            if index == 0:
                depth_rgb = first_depth
            else:
                depth_rgb = _crop_to_even_dimensions(_vkitti_depth_to_depth_anything_like(_load_depth_array(frame_path)))
                if depth_rgb.shape[:2] != (height, width):
                    raise ValueError(
                        f"All GT-depth frames must share resolution {(width, height)}; "
                        f"got {depth_rgb.shape[1]}x{depth_rgb.shape[0]} for {frame_path}"
                    )

            out_frame = av.VideoFrame.from_ndarray(depth_rgb, format="rgb24")
            for packet in stream.encode(out_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    print(
        "Created GT-depth conditioning video: "
        f"{output} from {len(depth_frames)} {dataset} depth frame(s) in {depth_dir}"
    )
    return output
