#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC_DIRS = (
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
)
for src_dir in PACKAGE_SRC_DIRS:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.rf_samplers import ContextFlowRFSolver2Sampler
from ltx_pipelines.inversion.schedule import CONTEXTFLOW_SIGMA_MIN


class ConstX0Model:
    """Fake LTX X0Model for a constant rectified-flow vector field.

    v(x, sigma) = c, so the LTX-style clean prediction is
    D(x, sigma) = x - sigma * c.
    """

    def __init__(self, c: torch.Tensor) -> None:
        self.c = c
        self.calls: list[tuple[float, tuple[int, ...]]] = []

    def __call__(self, video, audio, perturbations):
        del audio, perturbations
        sigma_view = video.timesteps.to(device=video.latent.device, dtype=video.latent.dtype)
        while sigma_view.dim() < video.latent.dim():
            sigma_view = sigma_view.unsqueeze(-1)
        c = self.c.to(device=video.latent.device, dtype=video.latent.dtype)
        self.calls.append((float(video.sigma.detach().cpu().reshape(-1)[0]), tuple(video.latent.shape)))
        return video.latent - sigma_view * c, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-form constant-flow debug test for ContextFlowRFSolver2Sampler.")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising intervals in the synthetic schedule.")
    parser.add_argument("--dtype", choices=("float32", "float64", "bf16"), default="float32")
    parser.add_argument("--step-parameterization", choices=("velocity", "x0"), default="velocity")
    parser.add_argument("--step-compute-dtype", choices=("float32", "float64", "bf16"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--first-order", action="store_true", help="Use ContextFlow first-order inversion/reconstruction updates.")
    parser.add_argument("--post-step-clamp", action="store_true", help="Enable post-step latent clamp ablation.")
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    dtype = {"float32": torch.float32, "float64": torch.float64, "bf16": torch.bfloat16}[args.dtype]
    step_compute_dtype = {"float32": torch.float32, "float64": torch.float64, "bf16": torch.bfloat16}[
        args.step_compute_dtype or args.dtype
    ]
    device = torch.device(args.device)
    torch.manual_seed(7)

    latent_shape = VideoLatentShape(batch=1, channels=3, frames=2, height=2, width=2)
    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=latent_shape,
        fps=24.0,
    )
    source = torch.randn(latent_shape.to_torch_shape(), device=device, dtype=dtype)

    # Build c in patchified token space because fake model receives Modality.latent.
    base_state = video_tools.create_initial_state(device=device, dtype=dtype, initial_latent=source)
    c = torch.linspace(0.05, 0.17, base_state.latent.numel(), device=device, dtype=dtype).reshape_as(base_state.latent)
    model = ConstX0Model(c)

    condition = BranchCondition(
        context=torch.zeros((1, 1, 4), device=device, dtype=dtype),
        conditionings=[],
    )
    timesteps = torch.linspace(1.0, CONTEXTFLOW_SIGMA_MIN, args.steps + 1, device=device, dtype=dtype)

    sampler = ContextFlowRFSolver2Sampler(
        video_tools=video_tools,
        use_reverse_pullback=False,
        post_step_clamp=args.post_step_clamp,
        first_order=args.first_order,
        step_parameterization=args.step_parameterization,
        step_compute_dtype=step_compute_dtype,
    )
    result = sampler.sample(
        transformer=model,
        source_latent=source,
        condition=condition,
        timesteps=timesteps,
    )

    diff = result.reconstructed_latent - source
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    print(f"step_parameterization: {args.step_parameterization}")
    print(f"state_dtype: {args.dtype}")
    print(f"step_compute_dtype: {args.step_compute_dtype or args.dtype}")
    print(f"timesteps: {timesteps.detach().cpu().tolist()}")
    print(f"model_calls: {len(model.calls)}")
    print(f"first_10_call_sigmas: {[round(s, 8) for s, _shape in model.calls[:10]]}")
    print(f"inverted_shape: {tuple(result.inverted_latent.shape)}")
    print(f"reconstructed_shape: {tuple(result.reconstructed_latent.shape)}")
    print(f"max_abs_reconstruction_error: {max_abs:.10g}")
    print(f"mean_abs_reconstruction_error: {mean_abs:.10g}")

    if max_abs > args.atol:
        raise SystemExit(f"FAIL: reconstruction error {max_abs:.10g} > atol {args.atol:.10g}")
    print(f"PASS: reconstruction error <= {args.atol:.10g}")


if __name__ == "__main__":
    main()
