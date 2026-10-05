"""Reference video conditioning for IC-LoRA inference."""

import json
from pathlib import Path

import torch

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape


class VideoConditionByReferenceLatent(ConditioningItem):
    """
    Conditions video generation on a reference video latent for IC-LoRA inference.
    IC-LoRAs are trained by concatenating reference (control signal) and target tokens,
    learning to attend across both. This class replicates that setup at inference by
    appending reference tokens to the latent sequence.
    IC-LoRAs can be trained with lower-resolution references than the target (e.g., 384px
    reference for 768px output) for efficiency and better generalization. The
    `downscale_factor` scales reference positions to match target coordinates, preserving
    the learned positional relationships. This must match the factor used during training
    (stored in LoRA metadata).
    To add attention masking, wrap with :class:`ConditioningItemAttentionStrengthWrapper`.
    Args:
        latent: Reference video latents [B, C, F, H, W]
        downscale_factor: Target/reference resolution ratio (e.g., 2 = half-resolution
            reference). Spatial positions are scaled by this factor.
        strength: Conditioning strength. 1.0 = full (reference kept clean),
            0.0 = none (reference denoised). Default 1.0.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        downscale_factor: int = 1,
        strength: float = 1.0,
        ref_position_quantize: int | None = None,
        ref_token_stride: int | None = None,
        ref_positions_log: str | None = None,
        source_strength: float | None = None,
        source_strength_routing: str | None = None,
    ):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength
        self.ref_position_quantize = ref_position_quantize
        self.ref_token_stride = ref_token_stride
        self.ref_positions_log = ref_positions_log
        self.source_strength = source_strength
        self.source_strength_routing = source_strength_routing
        if ref_position_quantize is not None and ref_token_stride is not None:
            raise ValueError("ref_position_quantize and ref_token_stride are mutually exclusive")
        for label, value in (
            ("ref_position_quantize", ref_position_quantize),
            ("ref_token_stride", ref_token_stride),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{label} must be a positive integer, got {value}")

    def _compute_positions(
        self,
        latent_tools: VideoLatentTools,
        shape: VideoLatentShape,
        *,
        downscale_factor: int,
    ) -> torch.Tensor:
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=shape,
            device=self.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps
        if downscale_factor != 1:
            positions[:, 1, ...] *= downscale_factor
            positions[:, 2, ...] *= downscale_factor
        return positions

    def _patch_grid_shape(self, latent_tools: VideoLatentTools, shape: VideoLatentShape) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = latent_tools.patchifier.patch_size
        if shape.frames % patch_t or shape.height % patch_h or shape.width % patch_w:
            raise ValueError(
                "reference latent shape is not divisible by patch size: "
                f"shape={shape}, patch_size={latent_tools.patchifier.patch_size}"
            )
        return shape.frames // patch_t, shape.height // patch_h, shape.width // patch_w

    def _filter_tokens_and_positions(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Hook for reference items that retain only a subset of the patch grid."""
        _ = grid_shape
        return tokens, positions, None

    def _write_positions_log(
        self,
        *,
        mode: str,
        q: int | None,
        positions: torch.Tensor,
        original_positions: torch.Tensor,
        tokens: torch.Tensor,
        latent_tools: VideoLatentTools,
        original_grid: tuple[int, int, int],
        selected_indices: torch.Tensor | None,
        canonical_positions: torch.Tensor | None,
        c6_gathered_positions: torch.Tensor | None,
    ) -> None:
        if self.ref_positions_log is None:
            return
        pos_cpu = positions.detach().to(device="cpu", dtype=torch.float32)
        spatial = pos_cpu[:, 1:3]
        unique_per_frame: list[int] = []
        frames = original_grid[0]
        if frames > 0:
            for frame_idx in range(frames):
                frame_mask = pos_cpu[:, 0, :, 0] == pos_cpu[:, 0, :, 0].unique(sorted=True)[min(frame_idx, pos_cpu[:, 0, :, 0].unique().numel() - 1)]
                coords = pos_cpu[0, 1:3, frame_mask[0], :].permute(1, 0, 2).reshape(-1, 4)
                unique_per_frame.append(int(torch.unique(coords, dim=0).shape[0]))
        sample_mappings = []
        if mode == "quantize" and q is not None:
            _, grid_h, grid_w = original_grid
            for fine_index in range(min(8, grid_h * grid_w)):
                h = fine_index // grid_w
                w = fine_index % grid_w
                token_index = h * grid_w + w
                sample_mappings.append(
                    {
                        "fine_token": [0, h, w],
                        "coarse_cell": [0, h // q, w // q],
                        "assigned_position": pos_cpu[0, :, token_index, :].tolist(),
                    }
                )
        payload = {
            "mode": mode,
            "q": q,
            "downscale_factor": self.downscale_factor,
            "source_strength": self.source_strength,
            "source_strength_routing": self.source_strength_routing,
            "reference_token_count": int(tokens.shape[1]),
            "latent_frames": int(original_grid[0]),
            "spatial_grid": [int(original_grid[1]), int(original_grid[2])],
            "position_shape": list(pos_cpu.shape),
            "spatial_position_min": spatial.amin(dim=(0, 2, 3)).tolist(),
            "spatial_position_max": spatial.amax(dim=(0, 2, 3)).tolist(),
            "unique_spatial_positions_per_latent_frame": unique_per_frame,
            "selected_token_indices_sample": (
                selected_indices[:16].detach().cpu().tolist() if selected_indices is not None else None
            ),
            "sample_mappings": sample_mappings,
            "max_abs_diff_vs_native_positions": float((positions - original_positions[:, :, : positions.shape[2]]).abs().max().item())
            if positions.shape[2] <= original_positions.shape[2]
            else None,
            "max_abs_diff_vs_canonical_downscale_positions": (
                float((positions - canonical_positions).abs().max().item()) if canonical_positions is not None else None
            ),
            "max_abs_diff_c6_gather_check": (
                float((positions - c6_gathered_positions).abs().max().item())
                if c6_gathered_positions is not None
                else None
            ),
        }
        path = Path(self.ref_positions_log).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Append reference video tokens with scaled or experimentally remapped positions."""
        tokens = latent_tools.patchifier.patchify(self.latent)
        reference_shape = VideoLatentShape.from_torch_shape(self.latent.shape)
        original_grid = self._patch_grid_shape(latent_tools, reference_shape)
        positions = self._compute_positions(
            latent_tools,
            reference_shape,
            downscale_factor=self.downscale_factor,
        )
        original_positions = positions
        mode = "native"
        q: int | None = None
        selected_indices: torch.Tensor | None = None
        canonical_positions: torch.Tensor | None = None
        c6_gathered_positions: torch.Tensor | None = None

        if self.ref_position_quantize is not None:
            mode = "quantize"
            q = int(self.ref_position_quantize)
            if self.downscale_factor != 1:
                raise ValueError("ref_position_quantize requires downscale_factor=1/full-res encoded content")
            frames, grid_h, grid_w = original_grid
            if grid_h % q or grid_w % q or reference_shape.height % q or reference_shape.width % q:
                raise ValueError(
                    f"ref_position_quantize={q} must divide reference latent/token grid; "
                    f"latent=({reference_shape.height},{reference_shape.width}), grid=({grid_h},{grid_w})"
                )
            coarse_shape = VideoLatentShape(
                batch=reference_shape.batch,
                channels=reference_shape.channels,
                frames=reference_shape.frames,
                height=reference_shape.height // q,
                width=reference_shape.width // q,
            )
            coarse_positions = self._compute_positions(latent_tools, coarse_shape, downscale_factor=q)
            coarse_grid = self._patch_grid_shape(latent_tools, coarse_shape)
            coarse_indices = []
            for f in range(frames):
                for h in range(grid_h):
                    for w in range(grid_w):
                        coarse_indices.append((f * coarse_grid[1] + (h // q)) * coarse_grid[2] + (w // q))
            index_tensor = torch.tensor(coarse_indices, device=self.latent.device, dtype=torch.long)
            positions = coarse_positions.index_select(2, index_tensor)
            c6_gathered_positions = positions.clone()
        elif self.ref_token_stride is not None:
            mode = "stride"
            q = int(self.ref_token_stride)
            if self.downscale_factor != 1:
                raise ValueError("ref_token_stride requires downscale_factor=1/full-res encoded content")
            frames, grid_h, grid_w = original_grid
            if grid_h % q or grid_w % q:
                raise ValueError(f"ref_token_stride={q} must divide token grid ({grid_h},{grid_w})")
            indices = []
            for f in range(frames):
                for h in range(0, grid_h, q):
                    for w in range(0, grid_w, q):
                        indices.append((f * grid_h + h) * grid_w + w)
            selected_indices = torch.tensor(indices, device=self.latent.device, dtype=torch.long)
            tokens = tokens.index_select(1, selected_indices)
            positions = positions.index_select(2, selected_indices)
            coarse_shape = VideoLatentShape(
                batch=reference_shape.batch,
                channels=reference_shape.channels,
                frames=reference_shape.frames,
                height=reference_shape.height // q,
                width=reference_shape.width // q,
            )
            canonical_positions = self._compute_positions(latent_tools, coarse_shape, downscale_factor=q)

        tokens, positions, keep_indices = self._filter_tokens_and_positions(tokens, positions, original_grid)
        if keep_indices is not None:
            if selected_indices is not None:
                raise ValueError("reference keep masks cannot be combined with ref_token_stride")
            selected_indices = keep_indices

        self._write_positions_log(
            mode=mode,
            q=q,
            positions=positions,
            original_positions=original_positions,
            tokens=tokens,
            latent_tools=latent_tools,
            original_grid=original_grid,
            selected_indices=selected_indices,
            canonical_positions=canonical_positions,
            c6_gathered_positions=c6_gathered_positions,
        )

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )


class MaskedReferenceVideoCondition(VideoConditionByReferenceLatent):
    """Append only reference tokens selected by a boolean ``(T,H,W)`` keep mask."""

    def __init__(
        self,
        *args,
        keep_mask: torch.Tensor,
        partition_name: str | None = None,
        partition_log_path: str | None = None,
        partition_log_reset: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if keep_mask.ndim != 3:
            raise ValueError(f"reference keep_mask must have shape (T,H,W), got {tuple(keep_mask.shape)}")
        self.keep_mask = keep_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        self.partition_name = partition_name
        self.partition_log_path = partition_log_path
        self.partition_log_reset = partition_log_reset

    def _filter_tokens_and_positions(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(self.keep_mask.shape) != grid_shape:
            raise ValueError(
                f"reference keep_mask shape {tuple(self.keep_mask.shape)} does not match patch grid {grid_shape}"
            )
        indices = torch.nonzero(self.keep_mask.reshape(-1), as_tuple=False).flatten().to(device=tokens.device)
        if indices.numel() == 0:
            raise ValueError("reference keep_mask selects no tokens")
        filtered_positions = positions.index_select(2, indices)
        self._write_partition_log(filtered_positions, indices, grid_shape)
        return tokens.index_select(1, indices), filtered_positions, indices

    def _write_partition_log(
        self, positions: torch.Tensor, indices: torch.Tensor, grid_shape: tuple[int, int, int]
    ) -> None:
        if self.partition_log_path is None or self.partition_name is None:
            return
        path = Path(self.partition_log_path).expanduser().resolve()
        payload = {}
        if path.exists() and not self.partition_log_reset:
            payload = json.loads(path.read_text())
        items = payload.setdefault("items", {})
        pos = positions.detach().float().cpu()
        mids = pos.mean(dim=-1)
        samples = []
        for local_index, flat_index in enumerate(indices[:10].detach().cpu().tolist()):
            frame = flat_index // (grid_shape[1] * grid_shape[2])
            rem = flat_index % (grid_shape[1] * grid_shape[2])
            row = rem // grid_shape[2]
            col = rem % grid_shape[2]
            samples.append({
                "grid_index": [frame, row, col],
                "bounds": pos[0, :, local_index].tolist(),
                "midpoint": mids[0, :, local_index].tolist(),
            })
        items[self.partition_name] = {
            "scale": self.downscale_factor,
            "source_strength": self.source_strength,
            "grid_shape": list(grid_shape),
            "kept_per_latent_frame": self.keep_mask.sum(dim=(1, 2)).tolist(),
            "total_tokens": int(indices.numel()),
            "spatial_bounds_min": pos[:, 1:3].amin(dim=(0, 2, 3)).tolist(),
            "spatial_bounds_max": pos[:, 1:3].amax(dim=(0, 2, 3)).tolist(),
            "samples": samples,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
