#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PRECOMPUTE_DEPTH_VIDEO_SCRIPT = SCRIPTS_DIR / "precompute_depth_video.py"
PACKAGE_SRC_DIRS = (
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
)
for src_dir in PACKAGE_SRC_DIRS:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput, QuantizationAction, QUANTIZATION_POLICIES
from ltx_pipelines.utils.media_io import encode_video

from depth_control_adapter import create_gt_depth_control_video

try:
    import cv2
except ImportError:
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LTX-2 IC-LoRA style transfer: preserve the motion/camera/structure of a "
            "reference video while steering appearance toward a reference image."
        )
    )
    parser.add_argument("--distilled-checkpoint-path", required=True, help="Path to the distilled LTX-2 checkpoint.")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Path to the spatial upsampler checkpoint.")
    parser.add_argument("--gemma-root", required=True, help="Path to the Gemma text-encoder directory.")
    parser.add_argument("--ic-lora-path", required=True, help="Path to the IC-LoRA weights.")
    parser.add_argument("--reference-video", required=True, help="Reference video whose motion and camera should be followed.")
    parser.add_argument(
        "--conditioning-video",
        default=None,
        help=(
            "Optional already-precomputed conditioning video. "
            "For a depth IC-LoRA, this can be a depth-map MP4; for edge, a Canny-edge MP4. "
            "If omitted, the script will use --reference-video directly for rgb mode "
            "or derive a control video from --reference-video for depth/edge mode."
        ),
    )
    parser.add_argument("--reference-image", required=True, help="Reference image whose style/appearance should influence the output.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--prompt",
        default=(
            "Follow the composition, timing, camera movement, and scene dynamics of the reference video, "
            "but render the result in the visual style, materials, colors, and identity implied by the reference image."
        ),
        help="Prompt to pair with the video and image conditioning.",
    )
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt text appended to the main prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output height. Defaults to the reference video height rounded down to a multiple of 64.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Defaults to the reference video width rounded down to a multiple of 64.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to generate. Defaults to the full reference video length.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Output frame rate. Defaults to the reference video FPS.",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index where the reference image is injected as image conditioning.",
    )
    parser.add_argument("--image-strength", type=float, default=1.0, help="Strength for the reference image conditioning.")
    parser.add_argument("--image-crf", type=int, default=33, help="CRF-like preprocessing parameter for image conditioning.")
    parser.add_argument("--video-strength", type=float, default=1.0, help="Strength for the reference video IC-LoRA conditioning.")
    parser.add_argument(
        "--conditioning-mode",
        choices=("rgb", "depth", "edge"),
        default="rgb",
        help="Interpretation of the conditioning video expected by the IC-LoRA.",
    )
    parser.add_argument(
        "--depth-model",
        default=None,
        help=(
            "Local path or model id for a Hugging Face depth-estimation model used when "
            "--conditioning-mode depth and --conditioning-video is not provided. "
            "Not required when --depth-backend video-depth-anything is used."
        ),
    )
    parser.add_argument(
        "--depth-backend",
        choices=("auto", "image", "video-depth-anything"),
        default="auto",
        help=(
            "Depth backend used to derive a conditioning video from --reference-video. "
            "'image' uses a per-frame model, 'video-depth-anything' uses a video-native model."
        ),
    )
    parser.add_argument(
        "--depth-cache-dir",
        default=None,
        help="Optional cache directory for loading the depth-estimation model.",
    )
    parser.add_argument(
        "--depth-device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used when precomputing depth videos.",
    )
    parser.add_argument(
        "--depth-use-fast",
        action="store_true",
        help="Request the fast image processor variant when supported by the image-depth backend.",
    )
    parser.add_argument(
        "--depth-output",
        default=None,
        help=(
            "Optional persistent output path for an auto-generated depth conditioning video. "
            "If the file already exists, it will be reused. "
            "If omitted, depth mode will cache to <output_dir>/conditioning/<reference_video_stem>_depth.mp4."
        ),
    )
    parser.add_argument(
        "--edge-output",
        default=None,
        help=(
            "Optional persistent output path for an auto-generated Canny edge conditioning video. "
            "If the file already exists, it will be reused. "
            "If omitted, edge mode will cache to <output_dir>/conditioning/<reference_video_stem>_edge.mp4."
        ),
    )
    parser.add_argument(
        "--edge-low-threshold",
        type=int,
        default=100,
        help="Lower Canny threshold used when auto-generating edge conditioning videos.",
    )
    parser.add_argument(
        "--edge-high-threshold",
        type=int,
        default=200,
        help="Upper Canny threshold used when auto-generating edge conditioning videos.",
    )
    parser.add_argument(
        "--accept-gt-depths",
        action="store_true",
        help="Use dataset-provided GT depth images instead of running a depth-estimation model.",
    )
    parser.add_argument("--gt-depth-dir", default=None, help="Root directory containing dataset GT-depth frames.")
    parser.add_argument(
        "--gt-depth-source-root",
        default=None,
        help="Root directory of the RGB/source dataset, used to map source paths to GT-depth paths.",
    )
    parser.add_argument(
        "--gt-depth-source-path",
        default=None,
        help="Current RGB/source sequence path, used to map to matching GT-depth frames.",
    )
    parser.add_argument(
        "--gt-depth-dataset",
        default=None,
        help="Dataset adapter name for GT-depth conversion. Defaults to --dataset from the wrapper.",
    )
    parser.add_argument(
        "--video-depth-anything-root",
        default=None,
        help="Path to a local Video Depth Anything checkout used when --depth-backend video-depth-anything.",
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
        help="Video Depth Anything encoder size.",
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
        help="Run Video Depth Anything in fp32 instead of fp16.",
    )
    parser.add_argument(
        "--conditioning-attention-strength",
        type=float,
        default=1.0,
        help="How strongly the reference video conditioning should influence attention, in [0, 1].",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Skip the upsampling/refinement stage for faster, lower-resolution output.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Let the text encoder enhance the prompt using the reference image as context.",
    )
    parser.add_argument(
        "--streaming-prefetch-count",
        type=int,
        default=None,
        help="Enable layer-streaming prefetching if supported by your setup.",
    )
    parser.add_argument(
        "--quiet-layer-streaming",
        action="store_true",
        help="Suppress warning-level layer-streaming memory logs.",
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action=QuantizationAction,
        nargs="+",
        metavar=("POLICY", "AMAX_PATH"),
        default=None,
        help=(
            f"Quantization policy: {', '.join(QUANTIZATION_POLICIES)}. "
            "Example: --quantization fp8-cast or --quantization fp8-scaled-mm /path/to/amax.json"
        ),
    )
    return parser.parse_args()


def read_video_metadata(video_path: str) -> tuple[int, int, int, float]:
    path = Path(video_path).expanduser().resolve()
    with av.open(str(path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {path}")

        width = int(video_stream.codec_context.width or video_stream.width)
        height = int(video_stream.codec_context.height or video_stream.height)

        fps_value = video_stream.average_rate or video_stream.base_rate or Fraction(25, 1)
        fps = float(fps_value)

        frames = video_stream.frames
        if not frames or frames <= 0:
            frames = 0
            for _packet in container.demux(video_stream):
                for _frame in _packet.decode():
                    frames += 1

    if frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")
    return width, height, frames, fps


def round_down_to_multiple(value: int, divisor: int) -> int:
    rounded = value - (value % divisor)
    if rounded <= 0:
        raise ValueError(f"Value {value} is too small to round down to a positive multiple of {divisor}")
    return rounded


def build_prompt(prompt: str, negative_prompt: str) -> str:
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    if not negative_prompt:
        return prompt
    return f"{prompt}\n\nNegative prompt: {negative_prompt}"


def default_depth_output_path(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).expanduser().resolve()
    reference_stem = Path(args.reference_video).expanduser().resolve().stem
    return output_path.parent / "conditioning" / f"{reference_stem}_depth.mp4"


def default_edge_output_path(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).expanduser().resolve()
    reference_stem = Path(args.reference_video).expanduser().resolve().stem
    return output_path.parent / "conditioning" / f"{reference_stem}_edge.mp4"


def is_readable_video_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False

    try:
        with av.open(str(path)) as container:
            return any(stream.type == "video" for stream in container.streams)
    except Exception:
        return False


def _depth_backend_requires_model(args: argparse.Namespace) -> bool:
    return args.depth_backend != "video-depth-anything"


def create_depth_video_from_rgb(
    rgb_video_path: str,
    output_path: str,
    depth_model_name_or_path: str | None,
    cache_dir: str | None = None,
    depth_backend: str = "auto",
    depth_device: str = "cuda",
    depth_use_fast: bool = False,
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
    if not PRECOMPUTE_DEPTH_VIDEO_SCRIPT.is_file():
        raise FileNotFoundError(f"Depth precompute script not found: {PRECOMPUTE_DEPTH_VIDEO_SCRIPT}")

    command = [
        sys.executable,
        str(PRECOMPUTE_DEPTH_VIDEO_SCRIPT),
        "--input",
        str(Path(rgb_video_path).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
        "--depth-backend",
        depth_backend,
        "--device",
        depth_device,
    ]
    if depth_model_name_or_path:
        command.extend(["--depth-model", depth_model_name_or_path])
    if cache_dir:
        command.extend(["--cache-dir", cache_dir])
    if depth_use_fast:
        command.append("--use-fast")
    if video_depth_anything_root:
        command.extend(["--video-depth-anything-root", video_depth_anything_root])
    if video_depth_anything_python:
        command.extend(["--video-depth-anything-python", video_depth_anything_python])
    if video_depth_anything_encoder != "vitl":
        command.extend(["--video-depth-anything-encoder", video_depth_anything_encoder])
    if video_depth_anything_metric:
        command.append("--video-depth-anything-metric")
    if video_depth_anything_input_size is not None:
        command.extend(["--video-depth-anything-input-size", str(video_depth_anything_input_size)])
    if video_depth_anything_max_res is not None:
        command.extend(["--video-depth-anything-max-res", str(video_depth_anything_max_res)])
    if video_depth_anything_max_len is not None:
        command.extend(["--video-depth-anything-max-len", str(video_depth_anything_max_len)])
    if video_depth_anything_target_fps is not None:
        command.extend(["--video-depth-anything-target-fps", str(video_depth_anything_target_fps)])
    if video_depth_anything_fp32:
        command.append("--video-depth-anything-fp32")

    subprocess.run(command, check=True)


def _predict_edge_frame(
    frame_rgb: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("Edge conditioning requested, but opencv-python/cv2 is not installed.")

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)
    return np.repeat(edges[:, :, None], 3, axis=2)


def create_edge_video_from_rgb(
    rgb_video_path: str,
    output_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> None:
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must be non-negative.")
    if high_threshold < low_threshold:
        raise ValueError("--edge-high-threshold must be greater than or equal to --edge-low-threshold.")

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

        for frame in input_container.decode(video=0):
            rgb_frame = frame.to_ndarray(format="rgb24")
            edge_rgb = _predict_edge_frame(rgb_frame, low_threshold, high_threshold)
            out_frame = av.VideoFrame.from_ndarray(edge_rgb, format="rgb24")
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)


def resolve_conditioning_video_path(args: argparse.Namespace) -> str:
    if args.accept_gt_depths:
        if args.conditioning_mode != "depth":
            raise ValueError("--accept-gt-depths requires --conditioning-mode depth")
        if args.conditioning_video:
            raise ValueError("--accept-gt-depths and --conditioning-video are mutually exclusive")
        missing = [
            name
            for name, value in (
                ("--gt-depth-dir", args.gt_depth_dir),
                ("--gt-depth-source-root", args.gt_depth_source_root),
                ("--gt-depth-source-path", args.gt_depth_source_path),
                ("--gt-depth-dataset", args.gt_depth_dataset),
            )
            if not value
        ]
        if missing:
            raise ValueError("--accept-gt-depths requires " + ", ".join(missing))

        depth_video_path = (
            Path(args.depth_output).expanduser().resolve() if args.depth_output else default_depth_output_path(args)
        )
        if is_readable_video_file(depth_video_path):
            print(f"Reusing existing GT-depth conditioning video: {depth_video_path}")
            return str(depth_video_path)

        print(f"GT-depth conditioning video not found at {depth_video_path}; generating it now.")
        create_gt_depth_control_video(
            dataset=args.gt_depth_dataset,
            gt_depth_root=args.gt_depth_dir,
            source_root=args.gt_depth_source_root,
            source_path=args.gt_depth_source_path,
            output_path=str(depth_video_path),
            fps=read_video_metadata(args.reference_video)[3],
            frame_cap=read_video_metadata(args.reference_video)[2],
        )
        return str(depth_video_path)

    if args.conditioning_video:
        conditioning_path = Path(args.conditioning_video).expanduser().resolve()
        generated_conditioning = False
        if not is_readable_video_file(conditioning_path):
            if args.conditioning_mode == "depth":
                if not args.depth_model:
                    raise ValueError(
                        f"conditioning video {conditioning_path} does not exist or is unreadable. "
                        "Provide --depth-model so the script can generate it automatically."
                    )
                print(f"Depth conditioning video not found at {conditioning_path}; generating it now.")
                create_depth_video_from_rgb(
                    rgb_video_path=str(Path(args.reference_video).expanduser().resolve()),
                    output_path=str(conditioning_path),
                    depth_model_name_or_path=args.depth_model,
                    cache_dir=args.depth_cache_dir,
                    depth_backend=args.depth_backend,
                    depth_device=args.depth_device,
                    depth_use_fast=args.depth_use_fast,
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
                generated_conditioning = True
            elif args.conditioning_mode == "edge":
                print(f"Edge conditioning video not found at {conditioning_path}; generating it now.")
                create_edge_video_from_rgb(
                    rgb_video_path=str(Path(args.reference_video).expanduser().resolve()),
                    output_path=str(conditioning_path),
                    low_threshold=args.edge_low_threshold,
                    high_threshold=args.edge_high_threshold,
                )
                generated_conditioning = True
            else:
                raise ValueError(f"conditioning video does not exist or is unreadable: {conditioning_path}")
        if not is_readable_video_file(conditioning_path):
            raise ValueError(f"conditioning video does not exist or is unreadable: {conditioning_path}")
        if not generated_conditioning:
            print(f"Reusing existing conditioning video: {conditioning_path}")
        return str(conditioning_path)

    if args.conditioning_mode == "rgb":
        return str(Path(args.reference_video).expanduser().resolve())

    if args.conditioning_mode == "edge":
        edge_video_path = (
            Path(args.edge_output).expanduser().resolve() if args.edge_output else default_edge_output_path(args)
        )
        if is_readable_video_file(edge_video_path):
            print(f"Reusing existing edge conditioning video: {edge_video_path}")
            return str(edge_video_path)

        print(f"Edge conditioning video not found at {edge_video_path}; generating it now.")
        create_edge_video_from_rgb(
            rgb_video_path=str(Path(args.reference_video).expanduser().resolve()),
            output_path=str(edge_video_path),
            low_threshold=args.edge_low_threshold,
            high_threshold=args.edge_high_threshold,
        )
        return str(edge_video_path)

    if _depth_backend_requires_model(args) and not args.depth_model:
        raise ValueError(
            "conditioning-mode=depth requires either --conditioning-video with a precomputed depth video "
            "or a depth backend configuration so the script can derive depth maps from --reference-video."
        )

    depth_video_path = (
        Path(args.depth_output).expanduser().resolve() if args.depth_output else default_depth_output_path(args)
    )
    if is_readable_video_file(depth_video_path):
        print(f"Reusing existing depth conditioning video: {depth_video_path}")
        return str(depth_video_path)

    print(f"Depth conditioning video not found at {depth_video_path}; generating it now.")
    create_depth_video_from_rgb(
        rgb_video_path=str(Path(args.reference_video).expanduser().resolve()),
        output_path=str(depth_video_path),
        depth_model_name_or_path=args.depth_model,
        cache_dir=args.depth_cache_dir,
        depth_backend=args.depth_backend,
        depth_device=args.depth_device,
        depth_use_fast=args.depth_use_fast,
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
    return str(depth_video_path)


def main() -> None:
    args = parse_args()

    if args.quiet_layer_streaming:
        logging.getLogger("ltx_core.layer_streaming").setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run LTX-2 IC-LoRA inference.")

    ref_width, ref_height, ref_num_frames, ref_fps = read_video_metadata(args.reference_video)

    width = args.width if args.width is not None else round_down_to_multiple(ref_width, 64)
    height = args.height if args.height is not None else round_down_to_multiple(ref_height, 64)
    num_frames = args.num_frames if args.num_frames is not None else ref_num_frames
    frame_rate = args.frame_rate if args.frame_rate is not None else ref_fps

    if width % 64 != 0 or height % 64 != 0:
        raise ValueError("height and width must both be divisible by 64 for the two-stage IC-LoRA pipeline.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not (0.0 <= args.conditioning_attention_strength <= 1.0):
        raise ValueError("conditioning_attention_strength must be between 0.0 and 1.0.")
    if args.image_frame_idx < 0 or args.image_frame_idx >= num_frames:
        raise ValueError(f"image-frame-idx must be in [0, {num_frames - 1}]")

    print(
        "Resolved video dimensions: "
        f"reference={ref_width}x{ref_height}, "
        f"pipeline_output={width}x{height}, "
        f"num_frames={num_frames}, "
        f"frame_rate={frame_rate:.3f}"
    )

    lora = LoraPathStrengthAndSDOps(
        path=str(Path(args.ic_lora_path).expanduser().resolve()),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )

    conditioning_video_path = resolve_conditioning_video_path(args)

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=[lora],
        device=torch.device("cuda"),
        quantization=args.quantization,
    )

    tiling_config = TilingConfig.default()
    video, audio = pipeline(
        prompt=build_prompt(args.prompt, args.negative_prompt),
        seed=args.seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=[
            ImageConditioningInput(
                path=str(Path(args.reference_image).expanduser().resolve()),
                frame_idx=args.image_frame_idx,
                strength=args.image_strength,
                crf=args.image_crf,
            )
        ],
        video_conditioning=[
            (
                conditioning_video_path,
                args.video_strength,
            )
        ],
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=args.conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        streaming_prefetch_count=args.streaming_prefetch_count,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=video,
        fps=int(round(frame_rate)),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=output_chunks,
    )

    print(f"Saved video to {output_path}")
    print(
        "Generation settings: "
        f"{width}x{height}, {num_frames} frames at {frame_rate:.3f} fps, "
        f"image strength={args.image_strength}, video strength={args.video_strength}, "
        f"conditioning mode={args.conditioning_mode}"
    )

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
