from contextlib import nullcontext

import torch

from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.inversion.attention.ace import ContextFlowACEPolicy
from ltx_pipelines.inversion.attention.controller import ContextFlowAttentionController
from ltx_pipelines.inversion.attention.feature_store import FeatureStore
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.schedule import LayerStepSchedule
from ltx_pipelines.inversion.trajectory import DualTrajectorySampler


class FakeTransformer:
    def __call__(self, video, audio, perturbations):
        assert audio is None
        context_scale = video.context.mean(dim=(1, 2), keepdim=True)
        return video.latent - video.timesteps + context_scale, None


def test_dual_trajectory_diverges_and_clears_cache() -> None:
    tools = VideoLatentTools(target_shape=VideoLatentShape(batch=1, channels=2, frames=2, height=1, width=1), fps=24.0, patchifier=__import__("ltx_core.components.patchifiers", fromlist=["VideoLatentPatchifier"]).VideoLatentPatchifier(patch_size=1))
    sampler = DualTrajectorySampler(video_tools=tools)
    feature_store = FeatureStore()
    controller = ContextFlowAttentionController(
        feature_store=feature_store,
        ace_policy=ContextFlowACEPolicy(schedule=LayerStepSchedule(active_layers=frozenset({0}), start_fraction=0.0, end_fraction=1.0)),
        active_layers={0},
    )
    initial_noise = torch.zeros(1, 2, 2, 1, 1)
    source_condition = BranchCondition(context=torch.zeros(1, 1, 2))
    target_condition = BranchCondition(context=torch.ones(1, 1, 2))
    timesteps = torch.tensor([1.0, 0.5, 0.0])
    source_latent, target_latent = sampler.sample(
        transformer=FakeTransformer(),
        initial_noise=initial_noise,
        source_condition=source_condition,
        target_condition=target_condition,
        timesteps=timesteps,
        attention_controller=controller,
    )
    assert source_latent.shape == target_latent.shape
    assert not torch.allclose(source_latent, target_latent)
    assert feature_store.is_empty()
