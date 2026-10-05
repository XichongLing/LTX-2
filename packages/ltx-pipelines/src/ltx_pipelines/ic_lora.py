import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import (
    ConditioningItem,
    ConditioningItemAttentionStrengthWrapper,
    ConditioningItemCorrespondenceBiasWrapper,
    MaskedReferenceVideoCondition,
    VideoConditionByReferenceLatent,
)
from ltx_core.loader import LoraPathStrengthAndSDOps, runtime_lora_scale
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
)
from ltx_core.model.video_vae import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    VideoEncoder,
    get_video_chunks_number,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.correspondence_mask import (
    build_masked_denoise_mask,
    downsample_disocclusion_mask_to_target_tokens,
)
from ltx_pipelines.reference_partition import Stage1ReferencePartitionConfig, load_reference_keep_mask
from ltx_pipelines.iclora_utils import (
    downsample_mask_video_to_latent,
    read_lora_reference_downscale_factor,
)
from ltx_pipelines.stage2_kv import Stage2KVAttentionController, Stage2KVCache, Stage2KVMode
from ltx_pipelines.stage2_noise import Stage2NoiseConfig, make_stage2_noiser
from ltx_pipelines.stage2_reference import (
    Stage2ReferenceAttentionController,
    Stage2ReferenceAttentionMode,
    Stage2ReferencePositionMode,
)
from ltx_pipelines.stage2_routing import (
    Stage2BranchMode,
    Stage2PredictionRecorder,
    downsample_semantic_mask_to_hard_target_tokens,
    downsample_semantic_mask_to_target_tokens,
    run_dual_trajectory_stage2,
    run_routed_stage2,
    save_stage2_input_cache,
    tensor_checksum,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.conditioning_schedules import (
    ConstantFirstFrameAttentionSchedule,
    ConstantSourceStrengthSchedule,
    FirstFrameAttentionEvaluation,
    FirstFrameAttentionSchedule,
    SourceStrengthEvaluation,
    SourceStrengthRouting,
    SourceStrengthSchedule,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser, SourceStrengthScheduledDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import (
    decode_video_by_frame,
    encode_video,
    load_image_and_preprocess,
    video_preprocess,
)
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class _RuntimeLoraDenoiser:
    """Enable attached runtime LoRA residuals for a legacy single-branch call."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        with runtime_lora_scale(transformer, 1.0):
            return self.delegate(transformer, video_state, audio_state, sigmas, step_index)


class _LegacyPredictionRecordingDenoiser:
    """Record legacy target predictions using the same schema as routed video mode."""

    def __init__(self, delegate: object, recorder: Stage2PredictionRecorder, target_count: int) -> None:
        self.delegate = delegate
        self.recorder = recorder
        self.target_count = target_count

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        video_result, audio_result = self.delegate(transformer, video_state, audio_state, sigmas, step_index)
        if video_result is not None and video_state is not None:
            target_input = video_state.latent[:, : self.target_count]
            target_x0 = video_result.denoised[:, : self.target_count]
            self.recorder.record(
                step_index=step_index,
                sigma=float(sigmas[step_index].detach().cpu()),
                shared_input=target_input,
                routed_x0=target_x0,
                image_x0=None,
                video_x0=target_x0,
                video_weight=torch.ones(
                    (*target_x0.shape[:2], 1),
                    device=target_x0.device,
                    dtype=target_x0.dtype,
                ),
            )
        return video_result, audio_result


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Both stages use distilled models for efficiency.
    """

    CONDITIONING_VIDEO_TILING = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
    )

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        stage_2_loras: list[LoraPathStrengthAndSDOps] | None = None,
        stage_2_runtime_loras: list[LoraPathStrengthAndSDOps] | None = None,
        reference_downscale_factor_override: int | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        stage_2_loras = stage_2_loras or []
        stage_2_runtime_loras = stage_2_runtime_loras or []
        if stage_2_loras and stage_2_runtime_loras:
            raise ValueError("Stage 2 cannot use fused and runtime LoRAs simultaneously")
        self.stage_2_ic_lora_enabled = bool(stage_2_loras)
        self.stage_2_runtime_lora_enabled = bool(stage_2_runtime_loras)

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage_1 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.stage_2 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(stage_2_loras),
            runtime_loras=tuple(stage_2_runtime_loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

        # Read reference downscale factor from LoRA metadata.
        # IC-LoRAs trained with low-resolution reference videos store this factor
        # so inference can resize reference videos to match training conditions.
        self.reference_downscale_factor = 1
        for lora in (*loras, *stage_2_loras, *stage_2_runtime_loras):
            scale = read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale
        if reference_downscale_factor_override is not None:
            if reference_downscale_factor_override < 1:
                raise ValueError("reference_downscale_factor_override must be at least 1")
            self.reference_downscale_factor = int(reference_downscale_factor_override)

    def __call__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        negative_prompt: str = "",
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        video_cfg_scale: float = 1.0,
        audio_cfg_scale: float = 1.0,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        attention_probe: object | None = None,
        reference_image_replace: set[int] | None = None,
        first_frame_attention_multiplier: float = 1.0,
        initial_video_latent: torch.Tensor | None = None,
        correspondence_mask: torch.Tensor | None = None,
        source_correspondence_bias: float = 0.0,
        source_correspondence_radius: int = 0,
        stage_2_masked_denoise_strength: float = 1.0,
        stage_2_conditioning_attention_strength: float = 1.0,
        source_strength_schedule: SourceStrengthSchedule | None = None,
        source_strength_routing: SourceStrengthRouting = SourceStrengthRouting.SYMMETRIC,
        stage_2_source_strength_schedule: SourceStrengthSchedule | None = None,
        source_strength_log: list[SourceStrengthEvaluation] | None = None,
        ref_position_quantize: int | None = None,
        ref_token_stride: int | None = None,
        ref_positions_log: str | None = None,
        stage_1_ref_partition_config: Stage1ReferencePartitionConfig | None = None,
        first_frame_attention_schedule: FirstFrameAttentionSchedule | None = None,
        stage_2_first_frame_attention_schedule: FirstFrameAttentionSchedule | None = None,
        first_frame_attention_log: list[FirstFrameAttentionEvaluation] | None = None,
        stage_2_branch_mode: Stage2BranchMode | str = Stage2BranchMode.LEGACY,
        stage_2_video_mix: float = 0.5,
        stage_2_routing_mask: torch.Tensor | None = None,
        stage_2_dress_video_contribution: float | None = None,
        stage_2_noise_seed: int | None = None,
        stage_2_noise_config: Stage2NoiseConfig | None = None,
        stage_2_prediction_dir: str | None = None,
        stage_2_kv_mode: Stage2KVMode | str = Stage2KVMode.NONE,
        stage_2_kv_strength: float = 1.0,
        stage_2_kv_layers: str = "all",
        stage_2_kv_cache_backend: str = "cpu",
        stage_2_kv_cache_dir: str | None = None,
        stage_2_dual_image_ic_lora: bool = True,
        stage_2_kv_include_inside_mask: bool = False,
        stage_2_kv_anchor_region: str = "outside",
        stage_2_appearance_reference: str | None = None,
        stage_2_appearance_reference_video: str | None = None,
        stage_2_appearance_reference_mask: torch.Tensor | None = None,
        stage_2_reference_attention_mode: Stage2ReferenceAttentionMode | str = Stage2ReferenceAttentionMode.NONE,
        stage_2_reference_position_mode: Stage2ReferencePositionMode | str = Stage2ReferencePositionMode.PRE_ROPE,
        stage_2_reference_kv_strength: float | None = None,
        stage_2_reference_kv_layers: str = "all",
        stage_2_reference_noise_seed: int | None = None,
        stage_2_reference_edit_conditioning: str = "none",
        cached_stage_2_video_latent: torch.Tensor | None = None,
        cached_stage_2_audio_latent: torch.Tensor | None = None,
        save_stage_2_input_path: str | None = None,
        stage_2_input_metadata: dict[str, object] | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """
        Generate video with IC-LoRA conditioning.
        Args:
            prompt: Text prompt for video generation.
            negative_prompt: Negative prompt used by CFG when cfg scale is > 1.
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            images: List of (path, frame_idx, strength) tuples for image conditioning.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            enhance_prompt: Whether to enhance the prompt using the text encoder.
            tiling_config: Optional tiling configuration for VAE decoding.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning attention.
                Controls how strongly the conditioning video influences the output.
                0.0 = ignore conditioning, 1.0 = full conditioning influence. Default 1.0.
                When conditioning_attention_mask is provided, the mask is multiplied by
                this strength before being passed to the conditioning items.
            skip_stage_2: If True, skip Stage 2 upsampling and refinement. Output will be
                at half resolution (height//2, width//2). Default is False.
            conditioning_attention_mask: Optional pixel-space attention mask with the same
                spatial-temporal dimensions as the input reference video. Shape should be
                (B, 1, F, H, W) or (1, 1, F, H, W) where F, H, W match the reference
                video's pixel dimensions. Values in [0, 1].
                The mask is downsampled to latent space using VAE scale factors (with
                causal temporal handling for the first frame), then multiplied by
                conditioning_attention_strength.
                When None (default): scalar conditioning_attention_strength is used
                directly.
            video_cfg_scale: CFG scale for video generation. 1.0 disables CFG.
            audio_cfg_scale: CFG scale for audio generation. 1.0 disables CFG.
            max_batch_size: Maximum batch size used inside guided denoising.
            reference_image_replace: Optional set of pixel-frame indices whose image
                conditionings should replace target latent frames in place. When None,
                preserves the historical default of replacing frame 0 only.
            first_frame_attention_multiplier: Attention multiplier for future target
                tokens attending to in-place frame-0 replacement tokens. 1.0 preserves
                current behavior; values >1.0 add positive attention-logit bias.
            correspondence_mask: Stage-1 pixel-space mask of shape (B,1,F,H,W).
                Values gate a directional target-query to RGB-source-key bias.
            source_correspondence_bias: Non-negative additive attention-logit bias
                applied at full mask confidence. Zero disables the feature.
            source_correspondence_radius: Radius, in source latent cells, around
                each aligned source key that receives the bias.
            stage_2_masked_denoise_strength: Per-token Stage-2 denoising strength in
                fully masked target regions. One preserves current behavior;
                zero locks those tokens to the upsampled Stage-1 latent. Intermediate
                values scale initial noise, model timesteps, and preservation blending.
            stage_2_conditioning_attention_strength: Attention strength for source-video
                reference tokens appended during Stage 2 when Stage 2 IC-LoRA is enabled.
            source_strength_schedule: Optional per-transformer-evaluation schedule for
                Stage-1 source-video attention. Constant 1.0 preserves the previous path.
            source_strength_routing: Which target/source attention blocks are scheduled.
                Symmetric scales both target->source and source->target.
            stage_2_source_strength_schedule: Optional schedule for Stage-2 source-video
                attention when Stage 2 IC-LoRA is enabled. Defaults to the Stage-1 schedule.
            source_strength_log: Optional list populated with source strength rows.
            first_frame_attention_schedule: Optional per-transformer-evaluation multiplier
                for later target tokens attending to in-place frame-0 target tokens.
            stage_2_first_frame_attention_schedule: Optional Stage-2 frame-0 multiplier
                schedule. Defaults to the Stage-1 frame-0 schedule.
            first_frame_attention_log: Optional list populated with frame-0 multiplier rows.
        Returns:
            Tuple of (video_iterator, audio_tensor).
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )
        if first_frame_attention_multiplier <= 0.0:
            raise ValueError(f"first_frame_attention_multiplier must be > 0.0, got {first_frame_attention_multiplier}")
        if source_correspondence_bias < 0.0:
            raise ValueError(f"source_correspondence_bias must be non-negative, got {source_correspondence_bias}")
        if source_correspondence_radius < 0:
            raise ValueError(f"source_correspondence_radius must be non-negative, got {source_correspondence_radius}")
        if not 0.0 <= stage_2_masked_denoise_strength <= 1.0:
            raise ValueError(
                "stage_2_masked_denoise_strength must be in [0, 1], "
                f"got {stage_2_masked_denoise_strength}"
            )
        if not 0.0 <= stage_2_conditioning_attention_strength <= 1.0:
            raise ValueError(
                "stage_2_conditioning_attention_strength must be in [0, 1], "
                f"got {stage_2_conditioning_attention_strength}"
            )
        if (self.stage_2_ic_lora_enabled or self.stage_2_runtime_lora_enabled) and not video_conditioning:
            raise ValueError("Stage 2 IC-LoRA requires at least one source-video conditioning")
        if stage_2_masked_denoise_strength < 1.0 and correspondence_mask is None:
            raise ValueError(
                "correspondence_mask is required when stage_2_masked_denoise_strength is below 1"
            )
        if source_correspondence_bias > 0.0 and correspondence_mask is None:
            raise ValueError("correspondence_mask is required when source_correspondence_bias is positive")
        if video_cfg_scale < 1.0:
            raise ValueError(f"video_cfg_scale must be >= 1.0, got {video_cfg_scale}")
        if audio_cfg_scale < 1.0:
            raise ValueError(f"audio_cfg_scale must be >= 1.0, got {audio_cfg_scale}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if (cached_stage_2_video_latent is None) != (cached_stage_2_audio_latent is None):
            raise ValueError("cached Stage-2 video and audio latents must be provided together")
        if cached_stage_2_video_latent is not None and skip_stage_2:
            raise ValueError("cached Stage-2 inputs cannot be used with skip_stage_2")
        if cached_stage_2_video_latent is not None and save_stage_2_input_path is not None:
            raise ValueError("cannot load and save a Stage-2 input cache in the same run")
        stage_2_branch_mode = Stage2BranchMode(stage_2_branch_mode)
        stage_2_kv_mode = Stage2KVMode(stage_2_kv_mode)
        stage_2_reference_attention_mode = Stage2ReferenceAttentionMode(stage_2_reference_attention_mode)
        stage_2_reference_position_mode = Stage2ReferencePositionMode(stage_2_reference_position_mode)
        if stage_2_reference_kv_strength is None:
            stage_2_reference_kv_strength = (
                1.0
                if stage_2_reference_attention_mode
                in {
                    Stage2ReferenceAttentionMode.REPLACE,
                    Stage2ReferenceAttentionMode.SOURCE_QK_REFERENCE_V,
                    Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV,
                }
                else 0.5
            )
        if stage_2_branch_mode.is_routed and not (
            self.stage_2_runtime_lora_enabled or self.stage_2_ic_lora_enabled
        ):
            raise ValueError("controlled Stage-2 branch modes require a Stage-2 IC-LoRA")
        if (
            stage_2_branch_mode.is_routed
            and self.stage_2_ic_lora_enabled
            and stage_2_branch_mode is not Stage2BranchMode.VIDEO
        ):
            raise ValueError("fused Stage-2 IC-LoRA is only valid for the single routed video branch")
        if stage_2_branch_mode is Stage2BranchMode.DUAL:
            if stage_2_routing_mask is None:
                raise ValueError("dual Stage-2 routing requires a routing mask")
            if stage_2_dress_video_contribution is None:
                raise ValueError("dual Stage-2 routing requires dress_video_contribution")
        elif stage_2_kv_mode is not Stage2KVMode.NONE:
            raise ValueError("Stage-2 KV guidance is only valid in dual branch mode")
        if stage_2_kv_include_inside_mask:
            stage_2_kv_anchor_region = "all"
        if stage_2_kv_anchor_region not in {"outside", "inside", "all"}:
            raise ValueError(
                f"stage_2_kv_anchor_region must be outside, inside, or all; got {stage_2_kv_anchor_region!r}"
            )
        if stage_2_branch_mode is not Stage2BranchMode.DUAL and stage_2_kv_include_inside_mask:
            raise ValueError("stage_2_kv_include_inside_mask is only valid in dual branch mode")
        if stage_2_branch_mode is not Stage2BranchMode.DUAL and stage_2_kv_anchor_region != "outside":
            raise ValueError("stage_2_kv_anchor_region is only valid in dual branch mode")
        if not 0.0 <= stage_2_kv_strength <= 1.0:
            raise ValueError(f"stage_2_kv_strength must be in [0,1], got {stage_2_kv_strength}")
        if stage_2_kv_cache_backend == "disk" and stage_2_kv_cache_dir is None:
            raise ValueError("stage_2_kv_cache_dir is required for the disk KV cache backend")
        reference_assets = [stage_2_appearance_reference, stage_2_appearance_reference_video]
        if sum(asset is not None for asset in reference_assets) > 1:
            raise ValueError("Stage-2 appearance reference image and video are mutually exclusive")
        reference_asset = next((asset for asset in reference_assets if asset is not None), None)
        if (reference_asset is None) != (stage_2_appearance_reference_mask is None):
            raise ValueError("A Stage-2 appearance reference image or video and mask must be provided together")
        reference_attention_enabled = stage_2_reference_attention_mode is not Stage2ReferenceAttentionMode.NONE
        if stage_2_reference_edit_conditioning not in {"none", "video"}:
            raise ValueError("stage_2_reference_edit_conditioning must be 'none' or 'video'")
        if reference_attention_enabled and stage_2_branch_mode is not Stage2BranchMode.DUAL:
            raise ValueError("Stage-2 reference attention is only valid in dual branch mode")
        if reference_attention_enabled and reference_asset is None:
            raise ValueError("Stage-2 reference attention requires an appearance reference image and mask")
        if reference_attention_enabled and stage_2_kv_mode is not Stage2KVMode.NONE:
            raise ValueError("Stage-2 reference attention cannot be combined with video-anchor KV guidance")
        if not reference_attention_enabled and reference_asset is not None:
            raise ValueError("Stage-2 appearance reference inputs require non-none reference attention")
        if not 0.0 <= stage_2_reference_kv_strength <= 1.0:
            raise ValueError("stage_2_reference_kv_strength must be in [0,1]")
        if (
            stage_2_reference_attention_mode
            in {
                Stage2ReferenceAttentionMode.REPLACE,
                Stage2ReferenceAttentionMode.SOURCE_QK_REFERENCE_V,
                Stage2ReferenceAttentionMode.SOURCE_Q_REFERENCE_KV,
            }
            and stage_2_reference_kv_strength != 1.0
        ):
            raise ValueError(
                f"Stage-2 reference {stage_2_reference_attention_mode.value} mode requires reference KV strength 1"
            )
        if stage_2_branch_mode is Stage2BranchMode.GLOBAL and not 0.0 <= stage_2_video_mix <= 1.0:
            raise ValueError("stage_2_video_mix must be in [0,1]")
        if stage_2_branch_mode is Stage2BranchMode.SPATIAL:
            if stage_2_routing_mask is None:
                raise ValueError("stage_2_routing_mask is required for spatial routing")
            if stage_2_dress_video_contribution is None or not 0.0 <= stage_2_dress_video_contribution <= 1.0:
                raise ValueError("stage_2_dress_video_contribution must be in [0,1]")
        if stage_2_branch_mode.is_routed and attention_probe is not None:
            raise ValueError("attention probing is not yet supported by routed Stage 2")
        stage_2_noise_seed = seed if stage_2_noise_seed is None else stage_2_noise_seed
        stage_2_noise_config = stage_2_noise_config or Stage2NoiseConfig()
        stage_2_reference_noise_seed = (
            stage_2_noise_seed + 3 if stage_2_reference_noise_seed is None else stage_2_reference_noise_seed
        )
        source_strength_schedule = source_strength_schedule or ConstantSourceStrengthSchedule(1.0)
        stage_2_source_strength_schedule = stage_2_source_strength_schedule or source_strength_schedule
        first_frame_attention_schedule = first_frame_attention_schedule or ConstantFirstFrameAttentionSchedule(1.0)
        stage_2_first_frame_attention_schedule = (
            stage_2_first_frame_attention_schedule or first_frame_attention_schedule
        )
        source_strength_routing = SourceStrengthRouting(source_strength_routing)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        video_negative_context, audio_negative_context = ctx_n.video_encoding, ctx_n.audio_encoding

        if cached_stage_2_video_latent is not None:
            upscaled_video_latent = cached_stage_2_video_latent.to(device=self.device, dtype=self.dtype)
            stage_2_audio_latent = cached_stage_2_audio_latent.to(device=self.device, dtype=self.dtype)
            logging.info("[IC-LoRA] Loaded cached Stage-2 video/audio inputs")
        else:
            # Stage 1: Initial low resolution video generation.
            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // 2,
                height=height // 2,
                fps=frame_rate,
            )
            logging.info(
                "[IC-LoRA] Stage 1 target pixel shape: "
                f"(B={stage_1_output_shape.batch}, C=3, F={stage_1_output_shape.frames}, "
                f"H={stage_1_output_shape.height}, W={stage_1_output_shape.width})"
            )

            stage_1_target_shape = VideoLatentShape.from_pixel_shape(stage_1_output_shape)
            stage_1_frame0_token_count = stage_1_target_shape.height * stage_1_target_shape.width

            # Encode conditionings using the video encoder block
            stage_1_conditionings, stage_1_source_token_counts = self.image_conditioner(
                lambda enc: self._create_conditionings(
                    images=images,
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=enc,
                    num_frames=num_frames,
                    conditioning_attention_strength=conditioning_attention_strength,
                    conditioning_attention_mask=conditioning_attention_mask,
                    # reference_image_replace=reference_image_replace,
                    first_frame_attention_multiplier=first_frame_attention_multiplier,
                    correspondence_mask=correspondence_mask,
                    source_correspondence_bias=source_correspondence_bias,
                    source_correspondence_radius=source_correspondence_radius,
                    ref_position_quantize=ref_position_quantize,
                    ref_token_stride=ref_token_stride,
                    ref_positions_log=ref_positions_log,
                    source_strength=source_strength_schedule.strength(
                        sigma=float(stage_1_sigmas[0].item()),
                        progress=0.0,
                        evaluation_index=0,
                        num_evaluations=stage_1_sigmas.numel() - 1,
                    ),
                    source_strength_routing=source_strength_routing.value,
                    reference_partition_config=stage_1_ref_partition_config,
                )
            )

            stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

            stage_1_denoiser = GuidedDenoiser(
                v_context=video_context,
                a_context=audio_context,
                video_guider=MultiModalGuider(
                    params=MultiModalGuiderParams(cfg_scale=video_cfg_scale),
                    negative_context=video_negative_context,
                ),
                audio_guider=MultiModalGuider(
                    params=MultiModalGuiderParams(cfg_scale=audio_cfg_scale),
                    negative_context=audio_negative_context,
                ),
            )
            stage_1_needs_scheduled_mask = (
                bool(stage_1_source_token_counts)
                or first_frame_attention_log is not None
                or not (
                    isinstance(first_frame_attention_schedule, ConstantFirstFrameAttentionSchedule)
                    and first_frame_attention_schedule.value == 1.0
                )
            )
            if stage_1_needs_scheduled_mask:
                stage_1_denoiser = SourceStrengthScheduledDenoiser(
                    stage_1_denoiser,
                    schedule=source_strength_schedule,
                    target_token_count=stage_1_target_shape.token_count(),
                    source_token_counts=stage_1_source_token_counts,
                    source_range_strengths=(
                        tuple(item.strength for item in stage_1_ref_partition_config.items)
                        if stage_1_ref_partition_config is not None
                        else None
                    ),
                    routing=source_strength_routing,
                    num_evaluations=stage_1_sigmas.numel() - 1,
                    sink=source_strength_log.append if source_strength_log is not None else None,
                    first_frame_schedule=first_frame_attention_schedule,
                    first_frame_token_count=stage_1_frame0_token_count,
                    first_frame_sink=first_frame_attention_log.append if first_frame_attention_log is not None else None,
                )

            video_state, audio_state = self.stage_1(
                denoiser=stage_1_denoiser,
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=stage_1_conditionings,
                    initial_latent=(
                        initial_video_latent.to(device=self.device, dtype=self.dtype)
                        if initial_video_latent is not None
                        else None
                    ),
                ),
                audio=ModalitySpec(
                    context=audio_context,
                ),
                loop=attention_probe.make_loop(
                    stage=1,
                    width=stage_1_output_shape.width,
                    height=stage_1_output_shape.height,
                    frames=stage_1_output_shape.frames,
                    fps=stage_1_output_shape.fps,
                    reference_token_counts=stage_1_source_token_counts,
                    reference_item_names=(
                        tuple(item.name for item in stage_1_ref_partition_config.items)
                        if stage_1_ref_partition_config is not None
                        else None
                    ),
                )
                if attention_probe is not None
                else None,
                max_batch_size=max_batch_size,
            )
            if video_state is not None:
                logging.info(f"[IC-LoRA] Stage 1 output latent shape: {tuple(video_state.latent.shape)}")

            if skip_stage_2:
                # Skip Stage 2: Decode directly from Stage 1 output at half resolution
                logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
                if video_state is not None:
                    logging.info(
                        "[IC-LoRA] Final decoded video will come from Stage 1 latent at half resolution: "
                        f"{tuple(video_state.latent.shape)}"
                    )
                decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
                decoded_audio = self.audio_decoder(audio_state.latent)
                return decoded_video, decoded_audio

            # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
            logging.info(f"[IC-LoRA] Stage 2 input latent shape: {tuple(upscaled_video_latent.shape)}")
            stage_2_audio_latent = audio_state.latent
            if save_stage_2_input_path is not None:
                save_stage2_input_cache(
                    save_stage_2_input_path,
                    video_latent=upscaled_video_latent,
                    audio_latent=stage_2_audio_latent,
                    metadata=stage_2_input_metadata or {},
                )
                logging.info("[IC-LoRA] Saved Stage-2 input cache to %s", save_stage_2_input_path)
        logging.info(f"[IC-LoRA] Stage 2 input latent shape: {tuple(upscaled_video_latent.shape)}")

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_target_shape = VideoLatentShape.from_pixel_shape(stage_2_output_shape)
        stage_2_frame0_token_count = stage_2_target_shape.height * stage_2_target_shape.width
        stage_2_denoise_mask = None
        if stage_2_masked_denoise_strength < 1.0:
            stage_2_denoise_mask = build_masked_denoise_mask(
                correspondence_mask.to(device=self.device),
                stage_2_target_shape,
                stage_2_masked_denoise_strength,
            )
            logging.info(
                "[IC-LoRA] Stage 2 masked per-token denoise strength: %.4f",
                stage_2_masked_denoise_strength,
            )
        logging.info(
            "[IC-LoRA] Stage 2 target pixel shape: "
            f"(B={stage_2_output_shape.batch}, C=3, F={stage_2_output_shape.frames}, "
            f"H={stage_2_output_shape.height}, W={stage_2_output_shape.width})"
        )
        if stage_2_branch_mode.is_routed:
            def _routed_conditionings(
                enc: VideoEncoder,
            ) -> tuple[list[ConditioningItem], list[ConditioningItem], tuple[int, ...], torch.Tensor | None]:
                image_items, _ = self._create_conditionings(
                    images=images,
                    video_conditioning=[],
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    num_frames=num_frames,
                )
                reference_latent = None
                if reference_asset is not None:
                    if stage_2_appearance_reference_video is not None:
                        reference_pixels = video_preprocess(
                            decode_video_by_frame(
                                path=stage_2_appearance_reference_video,
                                frame_cap=num_frames,
                                device=self.device,
                            ),
                            stage_2_output_shape.height,
                            stage_2_output_shape.width,
                            self.dtype,
                            self.device,
                        )
                        if reference_pixels.shape[2] != num_frames:
                            raise ValueError(
                                "Stage-2 appearance reference video must contain at least "
                                f"{num_frames} frames, decoded {reference_pixels.shape[2]}"
                            )
                    else:
                        reference_pixels = load_image_and_preprocess(
                            stage_2_appearance_reference,
                            stage_2_output_shape.height,
                            stage_2_output_shape.width,
                            self.dtype,
                            self.device,
                        )
                    reference_latent = enc.tiled_encode(
                        reference_pixels,
                        tiling_config=self.CONDITIONING_VIDEO_TILING,
                    )
                    logging.info(
                        "[Stage2 reference] Encoded appearance reference latent shape: %s",
                        tuple(reference_latent.shape),
                    )
                if stage_2_branch_mode is Stage2BranchMode.IMAGE:
                    return image_items, [], (), reference_latent
                video_items, source_counts = self._create_conditionings(
                    images=[],
                    video_conditioning=video_conditioning,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    num_frames=num_frames,
                    conditioning_attention_strength=stage_2_conditioning_attention_strength,
                    conditioning_attention_mask=conditioning_attention_mask,
                )
                return image_items, video_items, source_counts, reference_latent

            (
                stage_2_image_conditionings,
                stage_2_video_conditionings,
                stage_2_source_token_counts,
                stage_2_reference_latent,
            ) = self.image_conditioner(_routed_conditionings)
            image_denoiser = SimpleDenoiser(video_context, audio_context)
            image_needs_schedule = not (
                isinstance(stage_2_first_frame_attention_schedule, ConstantFirstFrameAttentionSchedule)
                and stage_2_first_frame_attention_schedule.value == 1.0
            )
            if image_needs_schedule:
                image_denoiser = SourceStrengthScheduledDenoiser(
                    image_denoiser,
                    schedule=ConstantSourceStrengthSchedule(1.0),
                    target_token_count=stage_2_target_shape.token_count(),
                    source_token_counts=(),
                    routing=source_strength_routing,
                    num_evaluations=stage_2_sigmas.numel() - 1,
                    first_frame_schedule=stage_2_first_frame_attention_schedule,
                    first_frame_token_count=stage_2_frame0_token_count,
                )
            video_denoiser = SimpleDenoiser(video_context, audio_context)
            video_denoiser = SourceStrengthScheduledDenoiser(
                video_denoiser,
                schedule=stage_2_source_strength_schedule,
                target_token_count=stage_2_target_shape.token_count(),
                source_token_counts=stage_2_source_token_counts,
                routing=source_strength_routing,
                num_evaluations=stage_2_sigmas.numel() - 1,
                sink=source_strength_log.append if source_strength_log is not None else None,
                first_frame_schedule=stage_2_first_frame_attention_schedule,
                first_frame_token_count=stage_2_frame0_token_count,
                first_frame_sink=first_frame_attention_log.append if first_frame_attention_log is not None else None,
            )
            video_edit_denoiser = None
            if stage_2_reference_edit_conditioning == "video":
                video_edit_denoiser = SourceStrengthScheduledDenoiser(
                    SimpleDenoiser(video_context, audio_context),
                    schedule=stage_2_source_strength_schedule,
                    target_token_count=stage_2_target_shape.token_count(),
                    source_token_counts=stage_2_source_token_counts,
                    routing=source_strength_routing,
                    num_evaluations=stage_2_sigmas.numel() - 1,
                    first_frame_schedule=stage_2_first_frame_attention_schedule,
                    first_frame_token_count=stage_2_frame0_token_count,
                )
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), stage_2_target_shape, frame_rate)
            audio_tools = AudioLatentTools(
                AudioPatchifier(patch_size=1), AudioLatentShape.from_video_pixel_shape(stage_2_output_shape)
            )
            reference_tools = None
            reference_hard_tokens = None
            if stage_2_reference_latent is not None:
                reference_shape = VideoLatentShape.from_torch_shape(stage_2_reference_latent.shape)
                reference_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), reference_shape, frame_rate)
                assert stage_2_appearance_reference_mask is not None
                reference_soft_tokens = downsample_semantic_mask_to_target_tokens(
                    stage_2_appearance_reference_mask.to(device=self.device), reference_tools
                )
                reference_hard_tokens = reference_soft_tokens >= 0.5
                if not reference_hard_tokens.any():
                    raise ValueError("Stage-2 appearance reference mask selects no latent tokens")
            routing_tokens = (
                downsample_semantic_mask_to_target_tokens(
                    stage_2_routing_mask.to(device=self.device), video_tools
                )
                if stage_2_routing_mask is not None
                else None
            )
            hard_routing_tokens = (
                downsample_semantic_mask_to_hard_target_tokens(
                    stage_2_routing_mask.to(device=self.device), video_tools
                )
                if stage_2_branch_mode is Stage2BranchMode.DUAL
                else None
            )
            recorder = (
                Stage2PredictionRecorder(
                    stage_2_prediction_dir,
                    metadata={
                        "mode": stage_2_branch_mode.value,
                        "video_mix": stage_2_video_mix,
                        "dress_video_contribution": stage_2_dress_video_contribution,
                        "noise_seed": stage_2_noise_seed,
                        "target_shape": list(upscaled_video_latent.shape),
                        "stage_2_input_checksum": tensor_checksum(upscaled_video_latent),
                        "stage_2_noise": stage_2_noise_config.manifest_dict(),
                        "stage_2_noise_diagnostics": stage_2_noise_config.diagnostics_sink,
                        "kv_mode": stage_2_kv_mode.value,
                        "dual_image_ic_lora": stage_2_dual_image_ic_lora,
                        "kv_include_inside_mask": stage_2_kv_anchor_region == "all",
                        "kv_anchor_region": stage_2_kv_anchor_region,
                        "kv_strength": stage_2_kv_strength,
                        "kv_layers": stage_2_kv_layers,
                        "kv_cache_backend": stage_2_kv_cache_backend,
                        "reference_attention_mode": stage_2_reference_attention_mode.value,
                        "reference_asset_type": ("video" if stage_2_appearance_reference_video is not None else "image"),
                        "reference_position_mode": stage_2_reference_position_mode.value,
                        "reference_image_checksum": (
                            hashlib.sha256(Path(reference_asset).expanduser().read_bytes()).hexdigest()
                            if reference_asset is not None
                            else None
                        ),
                        "reference_kv_strength": stage_2_reference_kv_strength,
                        "reference_kv_layers": stage_2_reference_kv_layers,
                        "reference_noise_seed": stage_2_reference_noise_seed,
                        "reference_edit_conditioning": stage_2_reference_edit_conditioning,
                        "anchor_source_token_counts": list(stage_2_source_token_counts),
                        "edit_source_token_counts": (
                            list(stage_2_source_token_counts)
                            if stage_2_reference_edit_conditioning == "video"
                            else []
                        ),
                        "reference_latent_checksum": (
                            tensor_checksum(stage_2_reference_latent) if stage_2_reference_latent is not None else None
                        ),
                        "reference_mask_checksum": (
                            tensor_checksum(stage_2_appearance_reference_mask)
                            if stage_2_appearance_reference_mask is not None
                            else None
                        ),
                    },
                    routing_mask=routing_tokens,
                    reference_mask=reference_hard_tokens,
                )
                if stage_2_prediction_dir is not None
                else None
            )
            with self.stage_2.model_context(video_tools=video_tools) as transformer:
                if stage_2_branch_mode is Stage2BranchMode.DUAL:
                    assert routing_tokens is not None
                    assert hard_routing_tokens is not None
                    assert stage_2_dress_video_contribution is not None
                    kv_cache = Stage2KVCache(
                        backend=stage_2_kv_cache_backend,
                        cache_dir=stage_2_kv_cache_dir,
                    )
                    kv_controller = Stage2KVAttentionController(
                        target_token_count=stage_2_target_shape.token_count(),
                        hard_mask=hard_routing_tokens,
                        mode=stage_2_kv_mode,
                        strength=stage_2_kv_strength,
                        layer_spec=stage_2_kv_layers,
                        cache=kv_cache,
                        include_inside_mask=stage_2_kv_include_inside_mask,
                        anchor_region=stage_2_kv_anchor_region,
                    )
                    reference_controller = None
                    reference_denoiser = None
                    if stage_2_reference_attention_mode is not Stage2ReferenceAttentionMode.NONE:
                        assert reference_tools is not None
                        assert reference_hard_tokens is not None
                        reference_controller = Stage2ReferenceAttentionController(
                            target_token_count=stage_2_target_shape.token_count(),
                            target_hard_mask=hard_routing_tokens,
                            reference_token_count=reference_tools.target_shape.token_count(),
                            reference_hard_mask=reference_hard_tokens,
                            mode=stage_2_reference_attention_mode,
                            position_mode=stage_2_reference_position_mode,
                            strength=stage_2_reference_kv_strength,
                            layer_spec=stage_2_reference_kv_layers,
                            cache=Stage2KVCache(
                                backend=stage_2_kv_cache_backend,
                                cache_dir=stage_2_kv_cache_dir,
                            ),
                        )
                        reference_denoiser = SimpleDenoiser(video_context, None)
                    video_state, audio_state = run_dual_trajectory_stage2(
                        transformer=transformer,
                        sigmas=stage_2_sigmas,
                        video_tools=video_tools,
                        audio_tools=audio_tools,
                        image_conditionings=stage_2_image_conditionings,
                        video_conditionings=stage_2_video_conditionings,
                        initial_video_latent=upscaled_video_latent,
                        initial_audio_latent=stage_2_audio_latent,
                        image_denoiser=image_denoiser,
                        video_denoiser=video_denoiser,
                        video_edit_denoiser=video_edit_denoiser,
                        noise_seed=stage_2_noise_seed,
                        routing_mask=routing_tokens,
                        dress_video_contribution=stage_2_dress_video_contribution,
                        kv_controller=kv_controller,
                        image_branch_ic_lora=stage_2_dual_image_ic_lora,
                        reference_tools=reference_tools if reference_controller is not None else None,
                        initial_reference_latent=(
                            stage_2_reference_latent if reference_controller is not None else None
                        ),
                        reference_denoiser=reference_denoiser,
                        reference_noise_seed=stage_2_reference_noise_seed,
                        reference_controller=reference_controller,
                        reference_edit_conditioning=stage_2_reference_edit_conditioning,
                        source_token_counts=stage_2_source_token_counts,
                        recorder=recorder,
                        video_denoise_mask=stage_2_denoise_mask,
                        stage_2_noise_config=stage_2_noise_config,
                        video_branch_runtime_lora=self.stage_2_runtime_lora_enabled,
                    )
                else:
                    video_state, audio_state = run_routed_stage2(
                        transformer=transformer,
                        sigmas=stage_2_sigmas,
                        video_tools=video_tools,
                        audio_tools=audio_tools,
                        image_conditionings=stage_2_image_conditionings,
                        video_conditionings=stage_2_video_conditionings,
                        initial_video_latent=upscaled_video_latent,
                        initial_audio_latent=stage_2_audio_latent,
                        image_denoiser=image_denoiser,
                        video_denoiser=video_denoiser,
                        mode=stage_2_branch_mode,
                        noise_seed=stage_2_noise_seed,
                        video_mix=stage_2_video_mix,
                        routing_mask=routing_tokens,
                        dress_video_contribution=stage_2_dress_video_contribution,
                        recorder=recorder,
                        video_denoise_mask=stage_2_denoise_mask,
                        stage_2_noise_config=stage_2_noise_config,
                    )
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        if self.stage_2_ic_lora_enabled or self.stage_2_runtime_lora_enabled:
            stage_2_conditionings, stage_2_source_token_counts = self.image_conditioner(
                lambda enc: self._create_conditionings(
                    images=images,
                    video_conditioning=video_conditioning,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    num_frames=num_frames,
                    conditioning_attention_strength=stage_2_conditioning_attention_strength,
                    conditioning_attention_mask=conditioning_attention_mask,
                )
            )
            logging.info(
                "[IC-LoRA] Stage 2 IC-LoRA enabled with %d source-video condition(s), "
                "conditioning attention strength %.4f",
                len(video_conditioning),
                stage_2_conditioning_attention_strength,
            )
        else:
            stage_2_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    dtype=self.dtype,
                    device=self.device,
                    # reference_image_replace=reference_image_replace,
                    # num_frames=num_frames,
                    # first_frame_attention_multiplier=first_frame_attention_multiplier,
                )
            )
            stage_2_source_token_counts = ()

        stage_2_denoiser = SimpleDenoiser(video_context, audio_context)
        stage_2_needs_scheduled_mask = (
            bool(stage_2_source_token_counts)
            or first_frame_attention_log is not None
            or not (
                isinstance(stage_2_first_frame_attention_schedule, ConstantFirstFrameAttentionSchedule)
                and stage_2_first_frame_attention_schedule.value == 1.0
            )
        )
        if stage_2_needs_scheduled_mask:
            stage_2_denoiser = SourceStrengthScheduledDenoiser(
                stage_2_denoiser,
                schedule=stage_2_source_strength_schedule,
                target_token_count=stage_2_target_shape.token_count(),
                source_token_counts=stage_2_source_token_counts,
                routing=source_strength_routing,
                num_evaluations=stage_2_sigmas.numel() - 1,
                sink=source_strength_log.append if source_strength_log is not None else None,
                first_frame_schedule=stage_2_first_frame_attention_schedule,
                first_frame_token_count=stage_2_frame0_token_count,
                first_frame_sink=first_frame_attention_log.append if first_frame_attention_log is not None else None,
            )
        if self.stage_2_runtime_lora_enabled:
            stage_2_denoiser = _RuntimeLoraDenoiser(stage_2_denoiser)
        if stage_2_prediction_dir is not None:
            legacy_recorder = Stage2PredictionRecorder(
                stage_2_prediction_dir,
                metadata={
                    "mode": "legacy",
                    "noise_seed": stage_2_noise_seed,
                    "target_shape": list(upscaled_video_latent.shape),
                    "stage_2_input_checksum": tensor_checksum(upscaled_video_latent),
                    "stage_2_noise": stage_2_noise_config.manifest_dict(),
                    "stage_2_noise_diagnostics": stage_2_noise_config.diagnostics_sink,
                    "source_token_counts": list(stage_2_source_token_counts),
                    "runtime_lora": self.stage_2_runtime_lora_enabled,
                },
                routing_mask=None,
            )
            stage_2_denoiser = _LegacyPredictionRecordingDenoiser(
                stage_2_denoiser,
                legacy_recorder,
                stage_2_target_shape.token_count(),
            )

        stage_2_video_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), stage_2_target_shape, frame_rate
        )
        stage_2_noiser = make_stage2_noiser(
            generator=generator,
            video_tools=stage_2_video_tools,
            reference_latent=upscaled_video_latent,
            config=stage_2_noise_config,
        )
        video_state, audio_state = self.stage_2(
            denoiser=stage_2_denoiser,
            sigmas=stage_2_sigmas,
            noiser=stage_2_noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                denoise_mask=stage_2_denoise_mask,
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=stage_2_audio_latent,
            ),
            loop=attention_probe.make_loop(
                stage=2,
                width=stage_2_output_shape.width,
                height=stage_2_output_shape.height,
                frames=stage_2_output_shape.frames,
                fps=stage_2_output_shape.fps,
            )
            if attention_probe is not None
            else None,
        )
        if video_state is not None:
            logging.info(f"[IC-LoRA] Stage 2 output latent shape: {tuple(video_state.latent.shape)}")

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    def _create_conditionings(  # noqa: PLR0913
        self,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
        reference_image_replace: set[int] | None = None,
        first_frame_attention_multiplier: float = 1.0,
        correspondence_mask: torch.Tensor | None = None,
        source_correspondence_bias: float = 0.0,
        source_correspondence_radius: int = 0,
        ref_position_quantize: int | None = None,
        ref_token_stride: int | None = None,
        ref_positions_log: str | None = None,
        source_strength: float | None = None,
        source_strength_routing: str | None = None,
        reference_partition_config: Stage1ReferencePartitionConfig | None = None,
    ) -> tuple[list[ConditioningItem], tuple[int, ...]]:
        """
        Create conditioning items for video generation.
        Args:
            conditioning_attention_strength: Scalar attention weight in [0, 1].
                If conditioning_attention_mask is also provided, the downsampled mask
                is multiplied by this strength. Otherwise this scalar is passed
                directly as the attention mask.
            conditioning_attention_mask: Optional pixel-space attention mask with shape
                (B, 1, F_pixel, H_pixel, W_pixel) matching the reference video's
                pixel dimensions. Downsampled to latent space with causal temporal
                handling, then multiplied by conditioning_attention_strength.
        Returns:
            List of conditioning items. IC-LoRA conditionings are appended last.
        """
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            # reference_image_replace=reference_image_replace,
            # num_frames=num_frames,
            # first_frame_attention_multiplier=first_frame_attention_multiplier,
        )
        source_token_counts: list[int] = []
        if ref_position_quantize is not None and ref_token_stride is not None:
            raise ValueError("--ref-position-quantize and --ref-token-stride are mutually exclusive")
        if (ref_position_quantize is not None or ref_token_stride is not None) and self.reference_downscale_factor != 1:
            raise ValueError(
                "Phase-2c reference token/position experiments require full-resolution reference encoding; "
                "pass --reference-downscale-factor-override 1"
            )

        if reference_partition_config is not None:
            if len(video_conditioning) != 1:
                raise ValueError("Stage-1 reference partition currently requires exactly one conditioning video")
            if ref_position_quantize is not None or ref_token_stride is not None:
                raise ValueError("reference partition cannot be combined with ref position/stride experiments")
            if conditioning_attention_mask is not None:
                raise ValueError("reference partition does not support a conditioning attention mask")
            if source_correspondence_bias > 0.0:
                raise ValueError("reference partition cannot be combined with source correspondence bias")
            video_path, video_strength = video_conditioning[0]
            for item_index, item in enumerate(reference_partition_config.items):
                if height % item.scale or width % item.scale:
                    raise ValueError(
                        f"Stage-1 dimensions ({height}x{width}) must be divisible by partition scale {item.scale}"
                    )
                ref_height = height // item.scale
                ref_width = width // item.scale
                frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=self.device)
                video = video_preprocess(frame_gen, ref_height, ref_width, self.dtype, self.device)
                encoded_video = video_encoder.tiled_encode(video, tiling_config=self.CONDITIONING_VIDEO_TILING)
                reference_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)
                keep_mask = load_reference_keep_mask(item.keep_mask_path)
                expected_shape = (reference_shape.frames, reference_shape.height, reference_shape.width)
                if tuple(keep_mask.shape) != expected_shape:
                    raise ValueError(
                        f"partition item {item.name!r} keep mask shape {tuple(keep_mask.shape)} "
                        f"does not match encoded reference grid {expected_shape}"
                    )
                source_token_counts.append(int(keep_mask.sum().item()))
                cond = MaskedReferenceVideoCondition(
                    latent=encoded_video,
                    downscale_factor=item.scale,
                    strength=video_strength,
                    keep_mask=keep_mask,
                    partition_name=item.name,
                    partition_log_path=(
                        str(reference_partition_config.log_path)
                        if reference_partition_config.log_path is not None
                        else None
                    ),
                    partition_log_reset=item_index == 0,
                    source_strength=item.strength,
                    source_strength_routing=source_strength_routing,
                )
                if conditioning_attention_strength != 1.0:
                    cond = ConditioningItemAttentionStrengthWrapper(
                        cond, attention_mask=conditioning_attention_strength
                    )
                conditionings.append(cond)
                logging.info(
                    "[IC-LoRA] Stage-1 partition item %s: scale=%d grid=%s kept=%d strength=%.4f",
                    item.name, item.scale, expected_shape, source_token_counts[-1], item.strength,
                )
            return conditionings, tuple(source_token_counts)

        # Calculate scaled dimensions for reference video conditioning.
        # IC-LoRAs trained with downscaled reference videos expect the same ratio at inference.
        scale = self.reference_downscale_factor
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
            )
        ref_height = height // scale
        ref_width = width // scale

        target_correspondence_mask = None
        if correspondence_mask is not None and source_correspondence_bias > 0.0:
            target_shape = VideoLatentShape.from_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=1.0)
            )
            target_correspondence_mask = downsample_disocclusion_mask_to_target_tokens(
                correspondence_mask.to(device=self.device), target_shape
            )

        for conditioning_index, (video_path, strength) in enumerate(video_conditioning):
            # Load video at scaled-down resolution (if scale > 1)
            frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=self.device)
            video = video_preprocess(frame_gen, ref_height, ref_width, self.dtype, self.device)
            logging.info(f"[IC-LoRA] Conditioning video pixel tensor shape: {tuple(video.shape)}")
            logging.info(
                "[IC-LoRA] Encoding conditioning video with tiled VAE encode "
                f"(ref size={ref_width}x{ref_height}, frames={video.shape[2]})"
            )
            encoded_video = video_encoder.tiled_encode(video, tiling_config=self.CONDITIONING_VIDEO_TILING)
            logging.info(f"[IC-LoRA] Conditioning video latent shape: {tuple(encoded_video.shape)}")
            reference_video_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)
            source_token_count = reference_video_shape.token_count()
            if ref_token_stride is not None:
                stride = int(ref_token_stride)
                if reference_video_shape.height % stride or reference_video_shape.width % stride:
                    raise ValueError(
                        f"--ref-token-stride {stride} must divide encoded reference latent shape "
                        f"({reference_video_shape.height}, {reference_video_shape.width})"
                    )
                source_token_count //= stride * stride
            source_token_counts.append(source_token_count)

            # Build attention_mask for ConditioningItemAttentionStrengthWrapper
            if conditioning_attention_mask is not None:
                # Downsample pixel-space mask to latent space, then scale by strength
                latent_mask = downsample_mask_video_to_latent(
                    mask=conditioning_attention_mask,
                    target_latent_shape=reference_video_shape,
                )
                attn_mask = latent_mask * conditioning_attention_strength
            elif conditioning_attention_strength < 1.0:
                # Use scalar strength only
                attn_mask = conditioning_attention_strength
            else:
                attn_mask = None

            cond = VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=scale,
                strength=strength,
                ref_position_quantize=ref_position_quantize,
                ref_token_stride=ref_token_stride,
                ref_positions_log=ref_positions_log,
                source_strength=source_strength,
                source_strength_routing=source_strength_routing,
            )
            if attn_mask is not None:
                cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
            if conditioning_index == 0 and target_correspondence_mask is not None:
                cond = ConditioningItemCorrespondenceBiasWrapper(
                    cond,
                    target_mask=target_correspondence_mask,
                    reference_shape=reference_video_shape,
                    logit_bias=source_correspondence_bias,
                    radius=source_correspondence_radius,
                )
                logging.info("[IC-LoRA] Added target-to-source correspondence bias to first video condition")
            conditionings.append(cond)

        if video_conditioning:
            logging.info("[IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings, tuple(source_token_counts)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    parser.add_argument(
        "--first-frame-attention-multiplier",
        type=float,
        default=1.0,
        help=(
            "Attention multiplier for future target tokens attending to in-place frame-0 "
            "replacement tokens. 1.0 preserves current behavior; values >1.0 add "
            "positive attention-logit bias."
        ),
    )
    args = parser.parse_args()

    # Load mask video if provided via --conditioning-attention-mask
    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,  # Stage 1 operates at half resolution
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        video_conditioning=args.video_conditioning,
        tiling_config=tiling_config,
        conditioning_attention_strength=conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        conditioning_attention_mask=conditioning_attention_mask,
        first_frame_attention_multiplier=args.first_frame_attention_multiplier,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].
    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.
    Returns:
        Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
    """
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
    mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
    # Normalise to [0, 1] — video_preprocess applies normalize_latent,
    # so undo that: values are in [-1, 1], remap to [0, 1]
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
