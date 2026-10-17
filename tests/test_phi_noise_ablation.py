from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def ablation_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "exp" / "phi_noise" / "run_stage2_phi_ablation.py"
    spec = importlib.util.spec_from_file_location("stage2_phi_ablation_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostic_gate_has_exact_nine_runs(ablation_module: ModuleType) -> None:
    specs = ablation_module.diagnostic_specs()
    assert len(specs) == 9
    assert len({spec.run_id for spec in specs}) == 9
    assert {spec.conditioning for spec in specs} >= {"image_only", "source_no_lora", "source_weak", "source_full"}
    assert any(spec.start_sigma == 0.8 for spec in specs)


def test_focused_suite_has_exact_59_unique_runs(ablation_module: ModuleType) -> None:
    specs = ablation_module.focused_specs()
    assert len(specs) == 59
    assert len({spec.run_id for spec in specs}) == 59
    assert sum(spec.noise_mode == "gaussian" and spec.start_sigma != 0.8 for spec in specs) == 4
    assert sum(spec.noise_mode == "phi" for spec in specs) == 36
    assert sum(spec.noise_mode == "raw_reference_coefficients" for spec in specs) == 9
    assert sum(spec.noise_mode == "matched_reference_coefficients" for spec in specs) == 9


def test_image_only_command_has_no_stage2_source_tokens_without_disabling_stage1_source(
    ablation_module: ModuleType, tmp_path: Path
) -> None:
    spec = ablation_module.diagnostic_specs()[0]
    command = ablation_module.build_command(
        spec,
        python="python",
        runner=Path("runner.py"),
        base_args=["--seed", "4"],
        output_dir=tmp_path,
        cache_path=tmp_path / "cache.safetensors",
        save_cache=True,
    )
    assert command[command.index("--stage-2-branch-mode") + 1] == "image"
    assert command[command.index("--stage-2-ic-lora-strength") + 1] == "0.0"
    assert "--no-source-video-conditioning" not in command
    assert "--save-stage-2-input" in command


def test_base_args_reject_runner_managed_options(ablation_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="runner-managed"):
        ablation_module._validate_base_args(["--output", "unsafe.mp4"])
