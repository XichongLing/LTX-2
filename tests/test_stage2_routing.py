from __future__ import annotations

import pytest
import torch
from torch import nn

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import VideoConditionByReferenceLatent
from ltx_core.loader.runtime_lora import RuntimeLoraFactors, RuntimeLoraLinear
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape
from ltx_pipelines.stage2_kv import Stage2KVAttentionController, Stage2KVCache
from ltx_pipelines.stage2_routing import (
    Stage2BranchMode,
    downsample_semantic_mask_to_target_tokens,
    load_stage2_input_cache,
    route_stage2_predictions,
    run_dual_trajectory_stage2,
    run_routed_stage2,
    save_stage2_input_cache,
)
from ltx_pipelines.utils.types import DenoisedLatentResult


def test_global_routing_endpoints_reduce_to_each_branch() -> None:
    image = torch.randn(1, 5, 3)
    video = torch.randn(1, 5, 3)
    at_image, image_weight = route_stage2_predictions(image, video, mode="global", video_mix=0.0)
    at_video, video_weight = route_stage2_predictions(image, video, mode="global", video_mix=1.0)
    assert torch.equal(at_image, image)
    assert torch.equal(at_video, video)
    assert torch.count_nonzero(image_weight) == 0
    assert torch.all(video_weight == 1)


def test_spatial_routing_uses_full_video_outside_and_requested_mix_inside() -> None:
    image = torch.zeros(1, 4, 2)
    video = torch.ones(1, 4, 2)
    mask = torch.tensor([[0.0, 1.0, 0.5, 1.0]])
    routed, weight = route_stage2_predictions(
        image,
        video,
        mode=Stage2BranchMode.SPATIAL,
        routing_mask=mask,
        dress_video_contribution=0.25,
    )
    expected = torch.tensor([1.0, 0.25, 0.625, 0.25])
    assert torch.allclose(weight[0, :, 0], expected)
    assert torch.allclose(routed[0, :, 0], expected)
    assert torch.allclose(routed[0, :, 1], expected)


def test_semantic_mask_uses_causal_temporal_mean_and_spatial_area() -> None:
    shape = VideoLatentShape(batch=1, channels=128, frames=2, height=1, width=1)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    mask = torch.tensor(
        [[[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]]
    )
    tokens = downsample_semantic_mask_to_target_tokens(mask, tools)
    assert tokens.shape == (1, 2)
    assert torch.allclose(tokens, torch.tensor([[1.0, 0.5]]))


def test_spatial_routing_validates_mask_shape_and_contribution() -> None:
    branch = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="mask shape"):
        route_stage2_predictions(
            branch,
            branch,
            mode="spatial",
            routing_mask=torch.zeros(1, 3),
            dress_video_contribution=0.0,
        )
    with pytest.raises(ValueError, match="dress_video_contribution"):
        route_stage2_predictions(branch, branch, mode="spatial", routing_mask=torch.zeros(1, 4))


def test_stage2_input_cache_roundtrip_and_metadata_validation(tmp_path) -> None:
    video = torch.randn(1, 4, 2, 3, 5, dtype=torch.bfloat16)
    audio = torch.randn(1, 8, 7, 2, dtype=torch.bfloat16)
    path = tmp_path / "stage2_input.safetensors"
    metadata = {"cache_version": 1, "stage_1_fingerprint": "abc"}
    save_stage2_input_cache(path, video_latent=video, audio_latent=audio, metadata=metadata)

    loaded_video, loaded_audio, manifest = load_stage2_input_cache(path, expected_metadata=metadata)
    assert torch.equal(loaded_video, video)
    assert torch.equal(loaded_audio, audio)
    assert manifest["video_shape"] == list(video.shape)
    assert manifest["audio_shape"] == list(audio.shape)

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_stage2_input_cache(path, expected_metadata={"stage_1_fingerprint": "different"})


class _BranchDenoiser:
    def __init__(self, *, expected_scale: float, video_offset: float, audio_offset: float, target_count: int) -> None:
        self.expected_scale = expected_scale
        self.video_offset = video_offset
        self.audio_offset = audio_offset
        self.target_count = target_count
        self.video_inputs = []
        self.full_video_inputs = []
        self.audio_inputs = []

    def __call__(self, transformer, video_state, audio_state, _sigmas, _step_index):
        adapter = next(module for module in transformer.modules() if isinstance(module, RuntimeLoraLinear))
        assert adapter.runtime_scale == self.expected_scale
        self.video_inputs.append(video_state.latent[:, : self.target_count].clone())
        self.full_video_inputs.append(video_state.latent.clone())
        self.audio_inputs.append(audio_state.latent.clone())
        return (
            DenoisedLatentResult(video_state.latent + self.video_offset),
            DenoisedLatentResult(audio_state.latent + self.audio_offset),
        )


def test_routed_sampler_shares_video_inputs_but_keeps_audio_histories_independent() -> None:
    video_shape = VideoLatentShape(batch=1, channels=2, frames=1, height=1, width=2)
    audio_shape = AudioLatentShape(batch=1, channels=2, frames=2, mel_bins=1)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), video_shape, fps=24.0)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
    target_count = video_shape.token_count()

    transformer = nn.Module()
    transformer.proj = RuntimeLoraLinear(
        nn.Linear(1, 1, bias=False),
        [RuntimeLoraFactors(torch.ones(1, 1), torch.ones(1, 1), 1.0)],
    )
    image_denoiser = _BranchDenoiser(
        expected_scale=0.0, video_offset=1.0, audio_offset=1.0, target_count=target_count
    )
    video_denoiser = _BranchDenoiser(
        expected_scale=1.0, video_offset=3.0, audio_offset=4.0, target_count=target_count
    )
    reference = VideoConditionByReferenceLatent(torch.zeros(1, 2, 1, 1, 1), strength=1.0)

    video_state, _audio_state = run_routed_stage2(
        transformer=transformer,
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        video_tools=video_tools,
        audio_tools=audio_tools,
        image_conditionings=[],
        video_conditionings=[reference],
        initial_video_latent=torch.zeros(video_shape.to_torch_shape()),
        initial_audio_latent=torch.zeros(audio_shape.to_torch_shape()),
        image_denoiser=image_denoiser,
        video_denoiser=video_denoiser,
        mode="global",
        noise_seed=123,
        video_mix=0.5,
    )

    assert video_state.latent.shape == video_shape.to_torch_shape()
    assert len(image_denoiser.video_inputs) == len(video_denoiser.video_inputs) == 2
    for image_input, video_input in zip(image_denoiser.video_inputs, video_denoiser.video_inputs, strict=True):
        assert torch.equal(image_input, video_input)
    assert torch.equal(image_denoiser.audio_inputs[0], video_denoiser.audio_inputs[0])
    assert not torch.equal(image_denoiser.audio_inputs[1], video_denoiser.audio_inputs[1])


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_function = lambda q, _k, _v, _heads: q
        self.masked_attention_function = lambda q, _k, _v, _heads, _mask: q


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _FakeAttention()


@pytest.mark.parametrize(("image_branch_ic_lora", "expected_image_scale"), [(False, 0.0), (True, 1.0)])
def test_dual_sampler_updates_anchor_independently_from_hybrid(
    image_branch_ic_lora: bool, expected_image_scale: float
) -> None:
    video_shape = VideoLatentShape(batch=1, channels=2, frames=1, height=1, width=2)
    audio_shape = AudioLatentShape(batch=1, channels=2, frames=2, mel_bins=1)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), video_shape, fps=24.0)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
    target_count = video_shape.token_count()

    transformer = nn.Module()
    transformer.proj = RuntimeLoraLinear(
        nn.Linear(1, 1, bias=False),
        [RuntimeLoraFactors(torch.ones(1, 1), torch.ones(1, 1), 1.0)],
    )
    transformer.velocity_model = nn.Module()
    transformer.velocity_model.transformer_blocks = nn.ModuleList([_FakeBlock()])
    image_denoiser = _BranchDenoiser(
        expected_scale=expected_image_scale, video_offset=1.0, audio_offset=1.0, target_count=target_count
    )
    video_denoiser = _BranchDenoiser(
        expected_scale=1.0, video_offset=3.0, audio_offset=4.0, target_count=target_count
    )
    reference = VideoConditionByReferenceLatent(torch.zeros(1, 2, 1, 1, 1), strength=1.0)
    controller = Stage2KVAttentionController(
        target_token_count=target_count,
        hard_mask=torch.tensor([[False, True]]),
        mode="none",
        strength=1.0,
        layer_spec="all",
        cache=Stage2KVCache(),
    )

    run_dual_trajectory_stage2(
        transformer=transformer,
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        video_tools=video_tools,
        audio_tools=audio_tools,
        image_conditionings=[],
        video_conditionings=[reference],
        initial_video_latent=torch.zeros(video_shape.to_torch_shape()),
        initial_audio_latent=torch.zeros(audio_shape.to_torch_shape()),
        image_denoiser=image_denoiser,
        video_denoiser=video_denoiser,
        noise_seed=123,
        routing_mask=torch.tensor([[0.0, 1.0]]),
        dress_video_contribution=0.0,
        kv_controller=controller,
        image_branch_ic_lora=image_branch_ic_lora,
    )

    assert torch.equal(image_denoiser.video_inputs[0], video_denoiser.video_inputs[0])
    assert not torch.equal(image_denoiser.video_inputs[1], video_denoiser.video_inputs[1])
    assert torch.equal(image_denoiser.audio_inputs[0], video_denoiser.audio_inputs[0])
    assert not torch.equal(image_denoiser.audio_inputs[1], video_denoiser.audio_inputs[1])


def test_dual_video_conditioned_edit_clones_source_tokens_but_diverges_independently() -> None:
    video_shape = VideoLatentShape(batch=1, channels=2, frames=1, height=1, width=2)
    audio_shape = AudioLatentShape(batch=1, channels=2, frames=2, mel_bins=1)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), video_shape, fps=24.0)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
    target_count = video_shape.token_count()

    transformer = nn.Module()
    transformer.proj = RuntimeLoraLinear(
        nn.Linear(1, 1, bias=False),
        [RuntimeLoraFactors(torch.ones(1, 1), torch.ones(1, 1), 1.0)],
    )
    transformer.velocity_model = nn.Module()
    transformer.velocity_model.transformer_blocks = nn.ModuleList([_FakeBlock()])
    edit_denoiser = _BranchDenoiser(
        expected_scale=1.0, video_offset=1.0, audio_offset=1.0, target_count=target_count
    )
    video_denoiser = _BranchDenoiser(
        expected_scale=1.0, video_offset=3.0, audio_offset=4.0, target_count=target_count
    )
    reference = VideoConditionByReferenceLatent(torch.zeros(1, 2, 1, 1, 1), strength=1.0)
    controller = Stage2KVAttentionController(
        target_token_count=target_count,
        hard_mask=torch.tensor([[False, True]]),
        mode="none",
        strength=1.0,
        layer_spec="all",
        cache=Stage2KVCache(),
    )

    run_dual_trajectory_stage2(
        transformer=transformer,
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        video_tools=video_tools,
        audio_tools=audio_tools,
        image_conditionings=[],
        video_conditionings=[reference],
        initial_video_latent=torch.zeros(video_shape.to_torch_shape()),
        initial_audio_latent=torch.zeros(audio_shape.to_torch_shape()),
        image_denoiser=edit_denoiser,
        video_denoiser=video_denoiser,
        video_edit_denoiser=edit_denoiser,
        noise_seed=123,
        routing_mask=torch.tensor([[0.0, 1.0]]),
        dress_video_contribution=0.0,
        kv_controller=controller,
        image_branch_ic_lora=True,
        reference_edit_conditioning="video",
        source_token_counts=(1,),
    )

    assert edit_denoiser.full_video_inputs[0].shape[1] > target_count
    assert torch.equal(edit_denoiser.full_video_inputs[0], video_denoiser.full_video_inputs[0])
    assert torch.equal(
        edit_denoiser.full_video_inputs[1][:, target_count:],
        video_denoiser.full_video_inputs[1][:, target_count:],
    )
    assert not torch.equal(edit_denoiser.video_inputs[1], video_denoiser.video_inputs[1])
    assert not torch.equal(edit_denoiser.audio_inputs[1], video_denoiser.audio_inputs[1])
