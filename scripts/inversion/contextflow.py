from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.inversion.attention.ace import ContextFlowACEPolicy
from ltx_pipelines.inversion.attention.controller import ContextFlowAttentionController
from ltx_pipelines.inversion.attention.feature_store import FeatureStore
from ltx_pipelines.inversion.instrumentation import ContextFlowInstrumentation
from ltx_pipelines.inversion.inversion import RectifiedFlowInverter
from ltx_pipelines.inversion.pipeline import ContextFlowPipeline
from ltx_pipelines.inversion.schedule import LayerStepSchedule
from ltx_pipelines.inversion.trajectory import DualTrajectorySampler
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.helpers import get_device
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.media_io import decode_video_from_file, encode_video, from_vae_range, load_image_and_preprocess, video_preprocess

checkpoint_path = "/home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors"
gemma_root = "/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized"
source_video_path = "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/source_video.mp4"
edited_first_frame_path = "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/flux/translated_first_frame.png"
USE_TILED_VIDEO_ENCODING = True
USE_SEQUENTIAL_PROMPT_ENCODING = True
DEFER_CONDITIONING_TO_GPU = True
PROMPT_OFFLOAD_MODE = OffloadMode.NONE
QUANTIZATION_KIND = QuantizationKind.FP8_CAST
ACE_INJECTION_MODE = "blend"
ACE_BLEND_ALPHA = 0.5
NUM_INFERENCE_STEPS = 10
ACE_LAYER_EXPERIMENTS = [
    ("early", {2, 4, 6, 8}),
    ("middle", {10, 12, 14, 16}),
    ("late", {18, 20, 22, 24}),
]
OUTPUT_ROOT = Path("/home/xichong-drz-inokko/experimental/LTX-2/scripts/inversion/outputs/contextflow_plane_10steps_blend_0.5_null_text_layers")


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
            for packet in container.demux(video_stream):
                for _frame in packet.decode():
                    frames += 1

    if frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")
    return width, height, frames, fps


def round_down_to_multiple(value: int, divisor: int) -> int:
    rounded = value - (value % divisor)
    if rounded <= 0:
        raise ValueError(f"Value {value} is too small to round down to a positive multiple of {divisor}")
    return rounded


def build_video_encoding_tiling_config(enabled: bool) -> TilingConfig | None:
    if not enabled:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
    )


def tensor_video_to_frames(video: torch.Tensor) -> torch.Tensor:
    if video.dim() == 4 and video.shape[-1] == 3:
        return video
    if video.dim() == 5 and video.shape[-1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(video.shape)}")
        return video.squeeze(0)
    if video.dim() == 5 and video.shape[1] == 3:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(video.shape)}")
        return video.squeeze(0).permute(1, 2, 3, 0)
    raise ValueError(f"Unsupported video tensor shape for saving: {tuple(video.shape)}")


def save_video_tensor(video: torch.Tensor, output_path: Path, fps: float, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(video).detach().cpu()
    if vae_range:
        frames = from_vae_range(frames)
    else:
        frames = frames.clamp(0, 1)
    encode_video(
        video=frames,
        fps=int(round(fps)),
        audio=None,
        output_path=str(output_path),
        video_chunks_number=1,
    )


def save_image_tensor(image: torch.Tensor, output_path: Path, *, vae_range: bool = False) -> None:
    frames = tensor_video_to_frames(image).detach().cpu()
    if vae_range:
        frames = from_vae_range(frames)
    else:
        frames = frames.clamp(0, 1)
    if frames.shape[0] != 1:
        raise ValueError(f"Expected a single frame image tensor, got {frames.shape[0]} frames")
    array = (frames[0].numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(output_path)


device = get_device()
preprocess_device = torch.device("cpu") if DEFER_CONDITIONING_TO_GPU else device
dtype = torch.bfloat16
preprocess_dtype = torch.float32 if DEFER_CONDITIONING_TO_GPU else dtype

ref_width, ref_height, _, ref_fps = read_video_metadata(source_video_path)
width = round_down_to_multiple(ref_width, 64)
height = round_down_to_multiple(ref_height, 64)
video_encoding_tiling_config = build_video_encoding_tiling_config(USE_TILED_VIDEO_ENCODING)
quantization = QUANTIZATION_KIND.to_policy(checkpoint_path)

# Keep conditioning tensors on CPU when deferred transfer is enabled so prompt encoding gets first access to VRAM.
frames = decode_video_from_file(source_video_path, device=preprocess_device)
source_video = video_preprocess(frames, height=height, width=width, dtype=preprocess_dtype, device=preprocess_device)
source_first_frame = source_video[:, :, :1]
edited_first_frame = load_image_and_preprocess(
    edited_first_frame_path,
    height=height,
    width=width,
    dtype=preprocess_dtype,
    device=preprocess_device,
)

pixel_shape = VideoPixelShape(
    batch=1,
    frames=source_video.shape[2],
    height=source_video.shape[3],
    width=source_video.shape[4],
    fps=ref_fps,
)
latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
video_tools = VideoLatentTools(
    patchifier=VideoLatentPatchifier(patch_size=1),
    target_shape=latent_shape,
    fps=ref_fps,
)

def run_layer_experiment(experiment_name: str, active_layers: set[int]) -> None:
    output_dir = OUTPUT_ROOT / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = LayerStepSchedule(
        active_layers=frozenset(active_layers),
        start_fraction=0.0,
        end_fraction=0.5,
    )
    controller = ContextFlowAttentionController(
        feature_store=FeatureStore(),
        ace_policy=ContextFlowACEPolicy(schedule=schedule, mode=ACE_INJECTION_MODE, blend_alpha=ACE_BLEND_ALPHA),
        active_layers=set(active_layers),
    )

    pipeline = ContextFlowPipeline(
        prompt_encoder=PromptEncoder(
            checkpoint_path,
            gemma_root,
            dtype,
            device,
            offload_mode=PROMPT_OFFLOAD_MODE,
        ),
        image_conditioner=ImageConditioner(checkpoint_path, dtype, device),
        diffusion_stage=DiffusionStage(checkpoint_path, dtype, device, quantization=quantization),
        video_decoder=VideoDecoder(checkpoint_path, dtype, device),
        inverter=RectifiedFlowInverter(video_tools=video_tools),
        trajectory_sampler=DualTrajectorySampler(video_tools=video_tools),
        attention_controller=controller,
        instrumentation=ContextFlowInstrumentation(),
        video_encoding_tiling_config=video_encoding_tiling_config,
        sequential_prompt_encoding=USE_SEQUENTIAL_PROMPT_ENCODING,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    print(f"Running experiment '{experiment_name}' with ACE layers {sorted(active_layers)}")
    with torch.inference_mode():
        result = pipeline(
            source_video=source_video,
            source_first_frame=source_first_frame,
            edited_first_frame=edited_first_frame,
            source_prompt="plane flying over landscape",
            target_prompt="same plane in the translated style",
            seed=42,
        )

    edited_video_path = output_dir / "edited_video.mp4"
    edited_video_no_ace_path = output_dir / "edited_video_no_ace.mp4"
    edited_video_with_ace_path = output_dir / "edited_video_with_ace.mp4"
    reconstructed_video_path = output_dir / "reconstructed_video.mp4"
    source_video_output_path = output_dir / "source_video.mp4"
    reference_first_frame_output_path = output_dir / "reference_first_frame.png"

    save_video_tensor(result.edited_video, edited_video_path, ref_fps)
    save_video_tensor(result.edited_video_no_ace, edited_video_no_ace_path, ref_fps)
    save_video_tensor(result.edited_video_with_ace, edited_video_with_ace_path, ref_fps)
    save_video_tensor(result.reconstructed_video, reconstructed_video_path, ref_fps)
    save_video_tensor(source_video, source_video_output_path, ref_fps, vae_range=True)
    save_image_tensor(edited_first_frame, reference_first_frame_output_path, vae_range=True)

    print(result.edited_video.shape, result.reconstructed_video.shape)
    print(f"Saved edited video to {edited_video_path}")
    print(f"Saved no-ACE edited video to {edited_video_no_ace_path}")
    print(f"Saved ACE edited video to {edited_video_with_ace_path}")
    print(f"Saved reconstructed video to {reconstructed_video_path}")
    print(f"Saved source video to {source_video_output_path}")
    print(f"Saved reference first frame to {reference_first_frame_output_path}")


torch.set_grad_enabled(False)
for experiment_name, active_layers in ACE_LAYER_EXPERIMENTS:
    run_layer_experiment(experiment_name, active_layers)
