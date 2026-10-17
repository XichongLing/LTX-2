from __future__ import annotations

import json
from pathlib import Path

import torch

from ltx_pipelines.stage2_routing import Stage2PredictionRecorder


def test_prediction_recorder_logs_first_model_response_only_once(tmp_path: Path) -> None:
    recorder = Stage2PredictionRecorder(tmp_path, metadata={"noise": "phi"}, routing_mask=None)
    model_input = torch.tensor([[[3.0, 4.0]]])
    x0 = torch.tensor([[[1.0, 2.0]]])
    weight = torch.zeros(1, 1, 1)
    recorder.record(
        step_index=0,
        sigma=0.5,
        shared_input=model_input,
        routed_x0=x0,
        image_x0=x0,
        video_x0=None,
        video_weight=weight,
    )
    recorder.record(
        step_index=1,
        sigma=0.25,
        shared_input=model_input,
        routed_x0=x0,
        image_x0=x0,
        video_x0=None,
        video_weight=weight,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    first = manifest["steps"][0]
    metrics = first["first_model_response"]["routed"]
    assert metrics["input_norm"] == 5.0
    assert metrics["x0_norm"] == torch.linalg.vector_norm(x0).item()
    assert metrics["velocity_norm"] == torch.linalg.vector_norm((model_input - x0) / 0.5).item()
    assert metrics["input_finite"] is True
    assert metrics["x0_finite"] is True
    assert metrics["velocity_finite"] is True
    assert "first_model_response" not in manifest["steps"][1]
