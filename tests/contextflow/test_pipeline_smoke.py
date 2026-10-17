from contextlib import contextmanager

import torch

from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.inversion.attention.controller import NoOpAttentionController
from ltx_pipelines.inversion.instrumentation import ContextFlowInstrumentation
from ltx_pipelines.inversion.inversion import RectifiedFlowInverter
from ltx_pipelines.inversion.pipeline import ContextFlowPipeline
from ltx_pipelines.inversion.trajectory import DualTrajectorySampler


class _PromptOutput:
    def __init__(self, value: float):
        self.video_encoding = torch.full((1, 1, 2), value)


class DummyPromptEncoder:
    def __call__(self, prompts):
        return [_PromptOutput(float(i)) for i, _ in enumerate(prompts)]


class DummyEncoder:
    def __call__(self, video):
        frames = 1 if video.shape[2] == 1 else 2
        return torch.ones(video.shape[0], 4, frames, 1, 1, device=video.device, dtype=video.dtype)


class DummyImageConditioner:
    def __call__(self, fn):
        return fn(DummyEncoder())


class DummyVideoDecoder:
    def __call__(self, latent):
        return iter([latent])


class DummyTransformer:
    def __call__(self, video, audio, perturbations):
        return video.latent - video.timesteps, None


class DummyDiffusionStage:
    @contextmanager
    def model_context(self, **kwargs):
        yield DummyTransformer()


def test_pipeline_smoke() -> None:
    tools = VideoLatentTools(
        patchifier=__import__("ltx_core.components.patchifiers", fromlist=["VideoLatentPatchifier"]).VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=1, channels=4, frames=2, height=1, width=1),
        fps=24.0,
    )
    pipeline = ContextFlowPipeline(
        prompt_encoder=DummyPromptEncoder(),
        image_conditioner=DummyImageConditioner(),
        diffusion_stage=DummyDiffusionStage(),
        video_decoder=DummyVideoDecoder(),
        inverter=RectifiedFlowInverter(video_tools=tools),
        trajectory_sampler=DualTrajectorySampler(video_tools=tools),
        attention_controller=NoOpAttentionController(),
        instrumentation=ContextFlowInstrumentation(),
    )
    source_video = torch.ones(1, 3, 9, 32, 32)
    source_first_frame = source_video[:, :, :1]
    edited_first_frame = source_video[:, :, :1] * 0.5
    result = pipeline(
        source_video=source_video,
        source_first_frame=source_first_frame,
        edited_first_frame=edited_first_frame,
        source_prompt="source",
        target_prompt="target",
        seed=0,
    )
    assert result.edited_video.shape == (1, 4, 2, 1, 1)
    assert result.reconstructed_video is not None
