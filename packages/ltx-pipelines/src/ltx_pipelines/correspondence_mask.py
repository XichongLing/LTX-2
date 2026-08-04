"""Mask sources and latent conversion for source-correspondence attention bias."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.media_io import decode_video_by_frame, resize_and_center_crop


@dataclass(frozen=True)
class CorrespondenceMaskBox:
    """A frame-inclusive box with normalized spatial coordinates."""

    start_frame: int
    end_frame: int
    x0: float
    y0: float
    x1: float
    y1: float

    def validate(self, num_frames: int) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame or self.end_frame >= num_frames:
            raise ValueError(
                f"Box frame range must satisfy 0 <= start <= end < {num_frames}, "
                f"got [{self.start_frame}, {self.end_frame}]"
            )
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(
                "Box coordinates must be normalized and satisfy "
                f"0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {(self.x0, self.y0, self.x1, self.y1)}"
            )


def build_later_frames_mask(
    *,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a full-spatial mask for every frame after frame zero."""
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("num_frames, height, and width must be positive")
    mask = torch.ones((1, 1, num_frames, height, width), device=device, dtype=torch.float32)
    mask[:, :, 0] = 0
    return mask


def build_box_mask(
    *,
    boxes: list[CorrespondenceMaskBox],
    num_frames: int,
    height: int,
    width: int,
    feather: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rasterize normalized spatiotemporal boxes into a canonical mask tensor."""
    if not boxes:
        raise ValueError("At least one box is required for correspondence mask mode 'boxes'")
    if feather < 0:
        raise ValueError(f"feather must be non-negative, got {feather}")

    mask = torch.zeros((1, 1, num_frames, height, width), device=device, dtype=torch.float32)
    for box in boxes:
        box.validate(num_frames)
        x0 = min(width - 1, int(box.x0 * width))
        y0 = min(height - 1, int(box.y0 * height))
        x1 = max(x0 + 1, min(width, int(np.ceil(box.x1 * width))))
        y1 = max(y0 + 1, min(height, int(np.ceil(box.y1 * height))))
        mask[:, :, box.start_frame : box.end_frame + 1, y0:y1, x0:x1] = 1

    if feather > 0:
        frames = rearrange(mask, "b c f h w -> (b f) c h w")
        kernel_size = 2 * feather + 1
        frames = torch.nn.functional.avg_pool2d(frames, kernel_size=kernel_size, stride=1, padding=feather)
        mask = rearrange(frames, "(b f) c h w -> b c f h w", b=1, f=num_frames)
    return mask


def _canonicalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().to(dtype=torch.float32, device="cpu")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif mask.dim() == 4:
        if mask.shape[-1] not in (1, 3, 4):
            raise ValueError(f"4-D mask tensors must have shape (F,H,W,C) with C in {{1,3,4}}; got {tuple(mask.shape)}")
    elif mask.dim() == 5:
        if mask.shape[0] != 1 or mask.shape[1] not in (1, 3, 4):
            raise ValueError(
                f"5-D mask tensors must have shape (1,C,F,H,W) with C in {{1,3,4}}; got {tuple(mask.shape)}"
            )
        mask = rearrange(mask, "1 c f h w -> f h w c")
    else:
        raise ValueError(
            f"Mask tensor must have shape (H,W), (F,H,W), (F,H,W,C), or (1,C,F,H,W); got {tuple(mask.shape)}"
        )

    if mask.shape[-1] > 1:
        mask = mask[..., :3].mean(dim=-1, keepdim=True)
    if mask.max().item() > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _load_mask_frames(path: Path, num_frames: int) -> torch.Tensor:
    if path.is_dir():
        image_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not image_paths:
            raise ValueError(f"No mask images found in directory {path}")
        frames = [
            torch.from_numpy(np.asarray(Image.open(image_path).convert("L")).copy()) for image_path in image_paths
        ]
        return _canonicalize_mask_tensor(torch.stack(frames))

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _canonicalize_mask_tensor(torch.from_numpy(np.load(path)))
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            payload = payload.get("mask", payload.get("tensor"))
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Expected a tensor or a dict containing 'mask' at {path}")
        return _canonicalize_mask_tensor(payload)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        frame = torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy())
        return _canonicalize_mask_tensor(frame)

    decoded = list(decode_video_by_frame(path=str(path), frame_cap=num_frames, device=torch.device("cpu")))
    if not decoded:
        raise ValueError(f"No frames decoded from correspondence mask video {path}")
    frames = torch.cat([frame.to(device="cpu") for frame in decoded], dim=0)
    return _canonicalize_mask_tensor(frames)


def load_file_mask(
    *,
    path: str | Path,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load a mask file, tensor, image sequence, or video and apply pipeline resize/crop."""
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"Correspondence mask does not exist: {mask_path}")

    frames = _load_mask_frames(mask_path, num_frames)
    if frames.shape[0] == 1 and num_frames > 1:
        frames = frames.expand(num_frames, -1, -1, -1)
    elif frames.shape[0] < num_frames:
        raise ValueError(f"Correspondence mask has {frames.shape[0]} frames, but generation needs {num_frames}")
    frames = frames[:num_frames]
    resized = resize_and_center_crop(frames, height, width)
    return resized.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def downsample_disocclusion_mask_to_target_tokens(
    mask: torch.Tensor,
    target_shape: VideoLatentShape,
) -> torch.Tensor:
    """Downsample a canonical pixel mask to target-token weights.

    Spatial dimensions use area pooling. Temporal dimensions follow the causal
    VAE layout—frame zero is isolated and later frames are grouped—with max
    pooling so brief disocclusions survive latent conversion.
    """
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape (B,1,F,H,W), got {tuple(mask.shape)}")
    if mask.shape[0] not in (1, target_shape.batch):
        raise ValueError(f"mask batch must be 1 or {target_shape.batch}, got {mask.shape[0]}")

    batch_size = mask.shape[0]
    spatial = torch.nn.functional.interpolate(
        rearrange(mask.to(dtype=torch.float32), "b 1 f h w -> (b f) 1 h w"),
        size=(target_shape.height, target_shape.width),
        mode="area",
    )
    spatial = rearrange(spatial, "(b f) 1 h w -> b 1 f h w", b=batch_size)

    first_frame = spatial[:, :, :1]
    if target_shape.frames == 1:
        latent_mask = first_frame
    else:
        remaining_frames = spatial.shape[2] - 1
        remaining_latents = target_shape.frames - 1
        if remaining_frames <= 0 or remaining_frames % remaining_latents != 0:
            raise ValueError(
                f"Pixel frames ({spatial.shape[2]}) are incompatible with target latent frames "
                f"({target_shape.frames}); expected F_pixel = 1 + K * (F_latent - 1)"
            )
        temporal_group = remaining_frames // remaining_latents
        rest = rearrange(
            spatial[:, :, 1:],
            "b 1 (f group) h w -> b 1 f group h w",
            f=remaining_latents,
            group=temporal_group,
        )
        rest = rest.amax(dim=3)
        latent_mask = torch.cat([first_frame, rest], dim=2)

    return rearrange(latent_mask, "b 1 f h w -> b (f h w)")


def build_masked_denoise_mask(
    mask: torch.Tensor,
    target_shape: VideoLatentShape,
    masked_strength: float,
) -> torch.Tensor:
    """Return a per-target-token Stage-2 denoise mask.

    Unmasked tokens remain at full denoising strength 1; fully masked tokens
    use ``masked_strength``. This consistently controls initial noising, model
    timesteps, and clean-latent blending throughout Stage 2.
    """
    if not 0.0 <= masked_strength <= 1.0:
        raise ValueError(f"masked_strength must be in [0, 1], got {masked_strength}")
    target_mask = downsample_disocclusion_mask_to_target_tokens(mask, target_shape)
    return 1.0 - target_mask * (1.0 - masked_strength)
