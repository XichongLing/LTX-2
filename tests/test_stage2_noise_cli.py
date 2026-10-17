from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from ltx_pipelines.utils.args import ImageConditioningInput


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "style_transfer_ltx23" / "run_ic_lora_style_transfer.py"
    spec = importlib.util.spec_from_file_location("phi_noise_cli_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phi_parameters_do_not_change_stage1_cache_fingerprint(tmp_path: Path) -> None:
    runner = _load_runner()
    assets = {}
    for name in ("checkpoint", "upsampler", "lora", "reference", "conditioning", "image"):
        path = tmp_path / name
        path.write_text(name)
        assets[name] = str(path)
    args = SimpleNamespace(
        seed=42,
        distilled_checkpoint_path=assets["checkpoint"],
        spatial_upsampler_path=assets["upsampler"],
        ic_lora_path=assets["lora"],
        reference_video=assets["reference"],
        conditioning_attention_strength=1.0,
        no_source_video_conditioning=False,
        stage_1_ic_lora_strength=1.0,
        stage_2_branch_mode="image",
        source_strength_routing="symmetric",
        source_strength_schedule="constant",
        source_strength=1.0,
        source_strength_values=None,
        source_strength_early=1.0,
        source_strength_late=1.0,
        source_strength_fade_start_sigma=1.0,
        source_strength_fade_end_sigma=0.0,
        first_frame_attention_schedule="constant",
        first_frame_attention_multiplier=1.0,
        first_frame_attention_values=None,
        first_frame_attention_early=1.0,
        first_frame_attention_late=1.0,
        first_frame_attention_fade_start_sigma=1.0,
        first_frame_attention_fade_end_sigma=0.0,
        stage_2_noise_mode="gaussian",
        stage_2_noise_area_fraction=0.05,
        stage_2_noise_gamma=5.0,
        stage_2_start_sigma=0.909375,
        phi_noise_temporal=False,
        phi_transform="temporal",
        phi_k=1,
        phi_gamma=30.0,
        phi_energy_scope="global",
        phi_exclude_frame0=False,
        phi_phase_source="source",
        phi_mask_mode="area_fraction",
        phi_area_fraction=0.05,
        phi_alpha=3,
        phi_source_latent=None,
    )
    kwargs = {
        "prompt": "prompt",
        "width": 384,
        "height": 256,
        "num_frames": 9,
        "frame_rate": 24.0,
        "image_conditionings": [ImageConditioningInput(assets["image"], 0, 1.0, 33)],
        "video_conditioning": [(assets["conditioning"], 1.0)],
    }
    gaussian = runner.build_stage_2_cache_metadata(args, **kwargs)
    args.stage_2_noise_mode = "phi"
    args.stage_2_noise_area_fraction = 0.10
    args.stage_2_noise_gamma = 30.0
    args.stage_2_start_sigma = 0.8
    phi = runner.build_stage_2_cache_metadata(args, **kwargs)
    assert gaussian["stage_1_fingerprint"] == phi["stage_1_fingerprint"]
    assert "stage_2_noise_mode" not in phi["stage_1_config"]
    assert phi["stage_1_config"]["spatial_upsampler"]["path"] == str(Path(assets["upsampler"]).resolve())

    args.phi_noise_temporal = True
    args.phi_transform = "spatial"
    args.phi_area_fraction = 0.10
    args.phi_k = 2
    stage1_phi = runner.build_stage_2_cache_metadata(args, **kwargs)
    assert stage1_phi["stage_1_fingerprint"] != gaussian["stage_1_fingerprint"]
    assert stage1_phi["stage_1_config"]["stage_1_phi_noise"]["enabled"] is True
