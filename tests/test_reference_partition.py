import json

import pytest
import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning import MaskedReferenceVideoCondition
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, VideoLatentShape
from ltx_pipelines.reference_partition import Stage1ReferencePartitionConfig
from ltx_pipelines.utils.conditioning_schedules import ConstantSourceStrengthSchedule, SourceStrengthRouting
from ltx_pipelines.utils.denoisers import SourceStrengthScheduledDenoiser
from ltx_pipelines.utils.types import DenoisedLatentResult


def _state(token_count: int, channels: int = 2) -> LatentState:
    latent = torch.zeros((1, token_count, channels))
    return LatentState(
        latent=latent,
        denoise_mask=torch.ones((1, token_count, 1)),
        positions=torch.zeros((1, 3, token_count, 2)),
        clean_latent=latent.clone(),
        attention_mask=None,
    )


def test_masked_reference_filters_every_token_aligned_tensor_in_fhw_order() -> None:
    shape = VideoLatentShape(batch=1, channels=2, frames=1, height=2, width=2)
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), shape, fps=24.0)
    reference = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    keep = torch.tensor([[[True, False], [False, True]]])
    condition = MaskedReferenceVideoCondition(reference, keep_mask=keep, downscale_factor=2)

    result = condition.apply_to(_state(4), tools)
    expected = tools.patchifier.patchify(reference)[:, [0, 3]]
    assert torch.equal(result.latent[:, 4:], expected)
    assert torch.equal(result.clean_latent[:, 4:], expected)
    assert result.positions.shape == (1, 3, 6, 2)
    assert result.denoise_mask.shape == (1, 6, 1)
    assert result.attention_mask is None


def test_partition_config_requires_ordered_scale2_scale4(tmp_path) -> None:
    for name, shape in (("K2.pt", (17, 6, 10)), ("K4.pt", (17, 3, 5))):
        torch.save({"mask": torch.ones(shape, dtype=torch.bool)}, tmp_path / name)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"items": [
        {"name": "scale2", "scale": 2, "keep_mask": "K2.pt"},
        {"name": "scale4", "scale": 4, "keep_mask": "K4.pt"},
    ]}))
    loaded = Stage1ReferencePartitionConfig.load(config)
    assert tuple(item.scale for item in loaded.items) == (2, 4)


class _Capture:
    def __init__(self):
        self.state = None

    def __call__(self, _transformer, video_state, audio_state, sigmas, step_index):
        self.state = video_state
        return DenoisedLatentResult(video_state.latent), None


def test_per_range_strengths_apply_only_to_corresponding_target_key_blocks() -> None:
    capture = _Capture()
    denoiser = SourceStrengthScheduledDenoiser(
        capture,
        schedule=ConstantSourceStrengthSchedule(1.0),
        target_token_count=4,
        source_token_counts=(2, 1),
        source_range_strengths=(4.0, 1.0),
        routing=SourceStrengthRouting.TARGET_QUERIES_ONLY,
    )
    state = _state(7)
    denoiser(object(), state, None, torch.tensor([1.0, 0.0]), 0)
    assert torch.equal(capture.state.attention_mask[:, :4, 4:6], torch.full((1, 4, 2), 4.0))
    assert torch.equal(capture.state.attention_mask[:, :4, 6:7], torch.ones((1, 4, 1)))
    assert torch.equal(capture.state.attention_mask[:, 4:7, :4], torch.ones((1, 3, 4)))


def test_per_range_strength_length_must_match_counts() -> None:
    with pytest.raises(ValueError, match="must match"):
        SourceStrengthScheduledDenoiser(
            _Capture(),
            schedule=ConstantSourceStrengthSchedule(1.0),
            target_token_count=4,
            source_token_counts=(2, 1),
            source_range_strengths=(1.0,),
        )


def test_build_partition_non_nested_scales_cover_every_stage1_cell(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    mask = np.zeros((384, 576), np.uint8)
    mask[100:300, 150:350] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    script = Path(__file__).resolve().parents[1] / "scripts/exp/two_scale/build_reference_partition.py"
    subprocess.run(
        [sys.executable, str(script), "--mask", str(tmp_path / "mask.png"), "--output-dir", str(tmp_path / "out"),
         "--num-frames", "9", "--stage1-width", "576", "--stage1-height", "384", "--fps", "24",
         "--inside-scale", "3", "--outside-scale", "2"],
        check=True, capture_output=True,
    )
    inside = torch.load(tmp_path / "out/K_inside.pt")["mask"]
    outside = torch.load(tmp_path / "out/K_outside.pt")["mask"]
    assert inside.shape == (2, 4, 6) and outside.shape == (2, 6, 9)
    inside_base = inside.repeat_interleave(3, dim=1).repeat_interleave(3, dim=2)
    outside_base = outside.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    assert torch.all(inside_base | outside_base)
    assert inside.any() and not torch.all(outside)
    summary = json.loads((tmp_path / "out/partition_summary.json").read_text())
    assert summary["nested_tiling"] is False
