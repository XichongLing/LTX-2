#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[2]
COMFYUI_ROOT = EXPERIMENTAL_ROOT / "ComfyUI"
COMFYUI_DEPTHANYTHING_ROOT = COMFYUI_ROOT / "custom_nodes" / "comfyui-depthanythingv2"
VIDEO_DEPTH_ANYTHING_ROOT = EXPERIMENTAL_ROOT / "Video-Depth-Anything"

for candidate in (COMFYUI_ROOT, COMFYUI_DEPTHANYTHING_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute a depth-map video from an RGB input video so depth estimation "
            "does not need to run in the same process as LTX-2 IC-LoRA inference."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the source RGB video.")
    parser.add_argument("--output", required=True, help="Where to save the depth MP4.")
    parser.add_argument(
        "--depth-model",
        default=None,
        help=(
            "Local path or Hugging Face model id for a depth-estimation model. "
            "Also supports ComfyUI DepthAnythingV2 .safetensors checkpoints. "
            "Not required when --depth-backend video-depth-anything is used."
        ),
    )
    parser.add_argument(
        "--depth-backend",
        choices=("auto", "image", "video-depth-anything"),
        default="auto",
        help=(
            "Depth backend to use. 'image' runs a per-frame depth estimator in this process. "
            "'video-depth-anything' shells out to an official Video Depth Anything checkout."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for the depth model.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for depth estimation.",
    )
    parser.add_argument(
        "--use-fast",
        action="store_true",
        help="Request the fast image processor variant when supported by the model.",
    )
    parser.add_argument(
        "--video-depth-anything-root",
        default=None,
        help=(
            "Path to a local Video Depth Anything checkout. "
            f"Defaults to {VIDEO_DEPTH_ANYTHING_ROOT} when present."
        ),
    )
    parser.add_argument(
        "--video-depth-anything-python",
        default=None,
        help="Python executable used to run Video Depth Anything. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--video-depth-anything-encoder",
        choices=("vits", "vitb", "vitl"),
        default="vitl",
        help="Video Depth Anything encoder size: vits, vitb, or vitl.",
    )
    parser.add_argument(
        "--video-depth-anything-metric",
        action="store_true",
        help="Use the metric Video Depth Anything checkpoint instead of the relative-depth checkpoint.",
    )
    parser.add_argument(
        "--video-depth-anything-input-size",
        type=int,
        default=None,
        help="Optional Video Depth Anything inference input size.",
    )
    parser.add_argument(
        "--video-depth-anything-max-res",
        type=int,
        default=None,
        help="Optional maximum input resolution for Video Depth Anything.",
    )
    parser.add_argument(
        "--video-depth-anything-max-len",
        type=int,
        default=None,
        help="Optional maximum video length for Video Depth Anything. Use -1 for no limit.",
    )
    parser.add_argument(
        "--video-depth-anything-target-fps",
        type=int,
        default=None,
        help="Optional target FPS for Video Depth Anything. Use -1 to keep the original FPS.",
    )
    parser.add_argument(
        "--video-depth-anything-fp32",
        action="store_true",
        help="Run Video Depth Anything in fp32 instead of its default fp16 inference mode.",
    )
    return parser.parse_args()


def _is_local_checkpoint(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path).expanduser()
    return path.is_file() and path.suffix == ".safetensors"


def _resolve_depth_backend(depth_backend: str, model_name_or_path: str | None) -> str:
    if depth_backend != "auto":
        return depth_backend
    return "image"


def _infer_comfy_depthanything_config(checkpoint_path: Path) -> tuple[dict[str, object], bool, float, torch.dtype]:
    model_name = checkpoint_path.name.lower()
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    if "vitg" in model_name:
        encoder = "vitg"
    elif "vitl" in model_name:
        encoder = "vitl"
    elif "vitb" in model_name:
        encoder = "vitb"
    elif "vits" in model_name:
        encoder = "vits"
    else:
        raise ValueError(
            f"Could not infer DepthAnythingV2 encoder from checkpoint name: {checkpoint_path.name}. "
            "Expected one of vits, vitb, vitl, vitg in the filename."
        )

    is_metric = "metric" in model_name
    max_depth = 20.0 if "hypersim" in model_name else 80.0
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    return model_configs[encoder], is_metric, max_depth, dtype


def _install_comfy_stubs() -> None:
    if "comfy.ops" in sys.modules and "comfy.ldm.modules.attention" in sys.modules:
        return

    comfy_module = sys.modules.setdefault("comfy", types.ModuleType("comfy"))

    ops_module = types.ModuleType("comfy.ops")
    manual_cast = types.SimpleNamespace(
        Linear=nn.Linear,
        Conv2d=nn.Conv2d,
    )
    ops_module.manual_cast = manual_cast
    sys.modules["comfy.ops"] = ops_module
    comfy_module.ops = ops_module

    ldm_module = sys.modules.setdefault("comfy.ldm", types.ModuleType("comfy.ldm"))
    comfy_module.ldm = ldm_module

    ldm_modules_module = sys.modules.setdefault("comfy.ldm.modules", types.ModuleType("comfy.ldm.modules"))
    ldm_module.modules = ldm_modules_module

    attention_module = types.ModuleType("comfy.ldm.modules.attention")

    def optimized_attention(q, k, v, num_heads, skip_reshape=False):  # noqa: ARG001
        attn = F.scaled_dot_product_attention(q, k, v)
        if skip_reshape:
            b, h, n, d = attn.shape
            return attn.transpose(1, 2).reshape(b, n, h * d)
        return attn

    attention_module.optimized_attention = optimized_attention
    sys.modules["comfy.ldm.modules.attention"] = attention_module
    ldm_modules_module.attention = attention_module


def _load_comfy_depthanything_checkpoint(checkpoint_path: str, device: str):
    _install_comfy_stubs()

    try:
        from safetensors.torch import load_file
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the local ComfyUI DepthAnythingV2 implementation. "
            "Make sure /home/xichong-drz-inokko/experimental/ComfyUI is present."
        ) from exc

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    model_config, is_metric, max_depth, dtype = _infer_comfy_depthanything_config(ckpt_path)

    if dtype == torch.float16:
        print(
            f"Using fp16 DepthAnything checkpoint {ckpt_path.name}. "
            "This matches your ComfyUI file, though ComfyUI notes fp16 can reduce depth quality."
        )

    model_kwargs = dict(model_config)
    if is_metric:
        model_kwargs["is_metric"] = True
        model_kwargs["max_depth"] = max_depth

    model = DepthAnythingV2(**model_kwargs)
    state_dict = load_file(str(ckpt_path))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return {
        "backend": "comfy_depthanything",
        "model": model,
        "dtype": dtype,
        "is_metric": is_metric,
    }


def _load_hf_depth_estimator(model_name_or_path: str, cache_dir: str | None, device: str, use_fast: bool):
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError("transformers is required to precompute the depth video.") from exc

    processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    model = model.to(device)
    model.eval()
    return {
        "backend": "huggingface",
        "processor": processor,
        "model": model,
    }


def load_depth_estimator(model_name_or_path: str, cache_dir: str | None, device: str, use_fast: bool):
    if _is_local_checkpoint(model_name_or_path):
        return _load_comfy_depthanything_checkpoint(model_name_or_path, device)
    return _load_hf_depth_estimator(model_name_or_path, cache_dir, device, use_fast)


def _run_video_depth_anything(
    rgb_video_path: str,
    output_path: str,
    repo_root: str | None,
    python_executable: str | None,
    encoder: str,
    metric: bool,
    input_size: int | None,
    max_res: int | None,
    max_len: int | None,
    target_fps: int | None,
    fp32: bool,
) -> None:
    root = Path(repo_root).expanduser().resolve() if repo_root else VIDEO_DEPTH_ANYTHING_ROOT
    script_path = root / "run.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            "Video Depth Anything backend requested, but run.py was not found. "
            f"Expected a checkout at {script_path}. "
            "Clone https://github.com/DepthAnything/Video-Depth-Anything or pass --video-depth-anything-root."
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_name = Path(rgb_video_path).expanduser().resolve().stem

    with tempfile.TemporaryDirectory(prefix=f"{output.stem}_vda_", dir=output.parent) as temp_dir:
        temp_output_dir = Path(temp_dir)
        command = [
            python_executable or sys.executable,
            str(script_path),
            "--input_video",
            str(Path(rgb_video_path).expanduser().resolve()),
            "--output_dir",
            str(temp_output_dir),
            "--encoder",
            encoder,
            "--grayscale",
        ]
        if metric:
            command.append("--metric")
        if input_size is not None:
            command.extend(["--input_size", str(input_size)])
        if max_res is not None:
            command.extend(["--max_res", str(max_res)])
        if max_len is not None:
            command.extend(["--max_len", str(max_len)])
        if target_fps is not None:
            command.extend(["--target_fps", str(target_fps)])
        if fp32:
            command.append("--fp32")

        print("Running Video Depth Anything:")
        print(" ".join(command))
        subprocess.run(command, check=True, cwd=str(root))

        produced_video = temp_output_dir / f"{input_name}_vis.mp4"
        if not produced_video.is_file():
            raise FileNotFoundError(
                "Video Depth Anything completed, but the expected visualization video was not created: "
                f"{produced_video}"
            )

        shutil.move(str(produced_video), str(output))


def predict_depth_frame(
    frame_rgb: np.ndarray,
    estimator: dict[str, object],
) -> np.ndarray:
    backend = estimator["backend"]

    if backend == "huggingface":
        image = Image.fromarray(frame_rgb)
        processor = estimator["processor"]
        model = estimator["model"]

        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).squeeze(0)
    elif backend == "comfy_depthanything":
        model = estimator["model"]
        dtype = estimator["dtype"]
        is_metric = bool(estimator["is_metric"])
        model_device = next(model.parameters()).device

        image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        _, orig_h, orig_w = image_tensor.shape

        proc_h = orig_h - (orig_h % 14)
        proc_w = orig_w - (orig_w % 14)
        if proc_h != orig_h or proc_w != orig_w:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(proc_h, proc_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor).unsqueeze(0).to(device=model_device, dtype=dtype)

        with torch.inference_mode():
            depth = model(image_tensor).squeeze(0).float()

        if depth.shape[-2:] != (orig_h, orig_w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        if is_metric:
            depth = 1.0 - depth
    else:
        raise ValueError(f"Unknown depth estimator backend: {backend}")

    depth = depth.float()
    depth = depth - depth.min()
    max_value = depth.max().item()
    if max_value > 0:
        depth = depth / max_value
    depth_uint8 = (depth.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    return np.repeat(depth_uint8[:, :, None], 3, axis=2)


def create_depth_video_from_rgb(
    rgb_video_path: str,
    output_path: str,
    depth_model_name_or_path: str | None,
    cache_dir: str | None = None,
    device: str = "cuda",
    use_fast: bool = False,
    depth_backend: str = "auto",
    video_depth_anything_root: str | None = None,
    video_depth_anything_python: str | None = None,
    video_depth_anything_encoder: str = "vitl",
    video_depth_anything_metric: bool = False,
    video_depth_anything_input_size: int | None = None,
    video_depth_anything_max_res: int | None = None,
    video_depth_anything_max_len: int | None = None,
    video_depth_anything_target_fps: int | None = None,
    video_depth_anything_fp32: bool = False,
) -> None:
    resolved_backend = _resolve_depth_backend(depth_backend, depth_model_name_or_path)
    if resolved_backend == "video-depth-anything":
        _run_video_depth_anything(
            rgb_video_path=rgb_video_path,
            output_path=output_path,
            repo_root=video_depth_anything_root,
            python_executable=video_depth_anything_python,
            encoder=video_depth_anything_encoder,
            metric=video_depth_anything_metric,
            input_size=video_depth_anything_input_size,
            max_res=video_depth_anything_max_res,
            max_len=video_depth_anything_max_len,
            target_fps=video_depth_anything_target_fps,
            fp32=video_depth_anything_fp32,
        )
        print(f"Saved depth video to {Path(output_path).expanduser().resolve()}")
        print("Processed video with Video Depth Anything")
        return

    if not depth_model_name_or_path:
        raise ValueError("--depth-model is required for the image depth backend.")

    estimator = load_depth_estimator(depth_model_name_or_path, cache_dir, device, use_fast)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with av.open(rgb_video_path) as input_container, av.open(str(output), mode="w") as output_container:
        input_stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
        if input_stream is None:
            raise ValueError(f"No video stream found in {rgb_video_path}")

        fps_value = input_stream.average_rate or input_stream.base_rate or Fraction(25, 1)
        fps = float(fps_value)
        width = int(input_stream.codec_context.width or input_stream.width)
        height = int(input_stream.codec_context.height or input_stream.height)

        output_stream = output_container.add_stream("libx264", rate=max(1, int(round(fps))))
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"

        frame_count = 0
        for frame in input_container.decode(video=0):
            rgb_frame = frame.to_ndarray(format="rgb24")
            depth_rgb = predict_depth_frame(rgb_frame, estimator)
            out_frame = av.VideoFrame.from_ndarray(depth_rgb, format="rgb24")
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)
            frame_count += 1

        for packet in output_stream.encode():
            output_container.mux(packet)

    print(f"Saved depth video to {output}")
    print(f"Processed {frame_count} frames on {device}")


def main() -> None:
    args = parse_args()
    create_depth_video_from_rgb(
        rgb_video_path=str(Path(args.input).expanduser().resolve()),
        output_path=str(Path(args.output).expanduser().resolve()),
        depth_model_name_or_path=args.depth_model,
        cache_dir=args.cache_dir,
        device=args.device,
        use_fast=args.use_fast,
        depth_backend=args.depth_backend,
        video_depth_anything_root=args.video_depth_anything_root,
        video_depth_anything_python=args.video_depth_anything_python,
        video_depth_anything_encoder=args.video_depth_anything_encoder,
        video_depth_anything_metric=args.video_depth_anything_metric,
        video_depth_anything_input_size=args.video_depth_anything_input_size,
        video_depth_anything_max_res=args.video_depth_anything_max_res,
        video_depth_anything_max_len=args.video_depth_anything_max_len,
        video_depth_anything_target_fps=args.video_depth_anything_target_fps,
        video_depth_anything_fp32=args.video_depth_anything_fp32,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
