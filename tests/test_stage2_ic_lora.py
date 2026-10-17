from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines import ic_lora


def _adapter(path: str, strength: float) -> LoraPathStrengthAndSDOps:
    return LoraPathStrengthAndSDOps(
        path=path,
        strength=strength,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )


def test_pipeline_configures_stage_2_loras_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    stage_loras: list[tuple[LoraPathStrengthAndSDOps, ...]] = []

    class FakeDiffusionStage:
        def __init__(
            self,
            *_args: object,
            loras: tuple[LoraPathStrengthAndSDOps, ...],
            **_kwargs: object,
        ) -> None:
            stage_loras.append(loras)

    def fake_component(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(ic_lora, "DiffusionStage", FakeDiffusionStage)
    for component_name in (
        "PromptEncoder",
        "ImageConditioner",
        "VideoUpsampler",
        "VideoDecoder",
        "AudioDecoder",
    ):
        monkeypatch.setattr(ic_lora, component_name, fake_component)
    monkeypatch.setattr(ic_lora, "read_lora_reference_downscale_factor", lambda _path: 2)

    stage_1_lora = _adapter("stage1.safetensors", 1.0)
    stage_2_lora = _adapter("stage2.safetensors", 0.5)
    pipeline = ic_lora.ICLoraPipeline(
        distilled_checkpoint_path="model.safetensors",
        spatial_upsampler_path="upsampler.safetensors",
        gemma_root="gemma",
        loras=[stage_1_lora],
        stage_2_loras=[stage_2_lora],
        device=torch.device("cpu"),
    )

    assert stage_loras == [(stage_1_lora,), (stage_2_lora,)]
    assert pipeline.stage_2_ic_lora_enabled is True
    assert pipeline.reference_downscale_factor == 2


def test_pipeline_preserves_empty_stage_2_loras_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stage_loras: list[tuple[LoraPathStrengthAndSDOps, ...]] = []

    class FakeDiffusionStage:
        def __init__(
            self,
            *_args: object,
            loras: tuple[LoraPathStrengthAndSDOps, ...],
            **_kwargs: object,
        ) -> None:
            stage_loras.append(loras)

    def fake_component(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(ic_lora, "DiffusionStage", FakeDiffusionStage)
    for component_name in (
        "PromptEncoder",
        "ImageConditioner",
        "VideoUpsampler",
        "VideoDecoder",
        "AudioDecoder",
    ):
        monkeypatch.setattr(ic_lora, component_name, fake_component)
    monkeypatch.setattr(ic_lora, "read_lora_reference_downscale_factor", lambda _path: 2)

    stage_1_lora = _adapter("stage1.safetensors", 1.0)
    pipeline = ic_lora.ICLoraPipeline(
        distilled_checkpoint_path="model.safetensors",
        spatial_upsampler_path="upsampler.safetensors",
        gemma_root="gemma",
        loras=[stage_1_lora],
        device=torch.device("cpu"),
    )

    assert stage_loras == [(stage_1_lora,), ()]
    assert pipeline.stage_2_ic_lora_enabled is False



def test_pipeline_configures_runtime_stage_2_lora_without_fusing(monkeypatch: pytest.MonkeyPatch) -> None:
    stages = []

    class FakeDiffusionStage:
        def __init__(self, *_args: object, loras=(), runtime_loras=(), **_kwargs: object) -> None:
            stages.append((loras, runtime_loras))

    def fake_component(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(ic_lora, "DiffusionStage", FakeDiffusionStage)
    for component_name in ("PromptEncoder", "ImageConditioner", "VideoUpsampler", "VideoDecoder", "AudioDecoder"):
        monkeypatch.setattr(ic_lora, component_name, fake_component)
    monkeypatch.setattr(ic_lora, "read_lora_reference_downscale_factor", lambda _path: 2)

    stage_1_lora = _adapter("stage1.safetensors", 1.0)
    runtime_lora = _adapter("runtime.safetensors", 0.75)
    pipeline = ic_lora.ICLoraPipeline(
        distilled_checkpoint_path="model.safetensors",
        spatial_upsampler_path="upsampler.safetensors",
        gemma_root="gemma",
        loras=[stage_1_lora],
        stage_2_runtime_loras=[runtime_lora],
        device=torch.device("cpu"),
    )

    assert stages == [((stage_1_lora,), ()), ((), (runtime_lora,))]
    assert pipeline.stage_2_ic_lora_enabled is False
    assert pipeline.stage_2_runtime_lora_enabled is True
    assert pipeline.reference_downscale_factor == 2

def test_single_sequence_wrapper_forwards_stage_2_controls() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "style_transfer_ltx23"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "translate_single_sequence_test",
            script_dir / "translate_single_sequence.py",
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    args = SimpleNamespace(
        distilled_checkpoint_path="model",
        spatial_upsampler_path="upsampler",
        gemma_root="gemma",
        ic_lora_path="adapter",
        output="out.mp4",
        prompt="prompt",
        negative_prompt="",
        seed=42,
        video_strength=1.0,
        conditioning_mode="rgb",
        conditioning_attention_strength=1.0,
        stage_2_ic_lora_strength=0.5,
        stage_2_conditioning_attention_strength=0.75,
        correspondence_mask_mode="later-frames",
        correspondence_mask_feather=0,
        source_correspondence_bias=0.0,
        source_correspondence_radius=0,
        stage_2_masked_denoise_strength=1.0,
        correspondence_mask_file=None,
        correspondence_mask_box=None,
        no_image_conditioning=False,
        image_frame_idx=0,
        image_strength=1.0,
        image_crf=33,
        rgb_strength=None,
        depth_strength=None,
        edge_strength=None,
        width=None,
        height=None,
        num_frames=None,
        frame_rate=None,
        skip_stage_2=False,
        enhance_prompt=False,
        quiet_layer_streaming=False,
        streaming_prefetch_count=None,
        attention_probe_output=None,
        quantization=None,
        reference_image_replace=None,
    )

    command = module.build_style_transfer_command(
        args,
        source_video="source.mp4",
        control_videos=["control.mp4"],
        reference_image="reference.png",
    )

    lora_flag_index = command.index("--stage-2-ic-lora-strength")
    attention_flag_index = command.index("--stage-2-conditioning-attention-strength")
    assert command[lora_flag_index + 1] == "0.5"
    assert command[attention_flag_index + 1] == "0.75"
    expected_stage_2_noise_options = {
        "--stage-2-noise-mode": "gaussian",
        "--stage-2-noise-transform": "spatial",
        "--stage-2-noise-mask-mode": "area_fraction",
        "--stage-2-noise-area-fraction": "0.05",
        "--stage-2-noise-alpha": "3",
        "--stage-2-noise-gamma": "5.0",
        "--stage-2-noise-phase-source": "stage1",
        "--stage-2-start-sigma": "0.909375",
    }
    for flag, expected in expected_stage_2_noise_options.items():
        assert command[command.index(flag) + 1] == expected
