#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.rf_samplers import ContextFlowRFSolver2Sampler
from ltx_pipelines.inversion.schedule import CONTEXTFLOW_SIGMA_MIN
from ltx_pipelines.utils.helpers import state_with_conditionings


class LinearX0Model:
    """Fake LTX X0Model for v(x, sigma) = a*x, returning D=x-sigma*v."""

    def __init__(self, a: float) -> None:
        self.a = float(a)
        self.calls: list[tuple[float, tuple[int, ...]]] = []

    def __call__(self, video, audio, perturbations):
        del audio, perturbations
        sigma_view = video.timesteps.to(device=video.latent.device, dtype=video.latent.dtype)
        while sigma_view.dim() < video.latent.dim():
            sigma_view = sigma_view.unsqueeze(-1)
        self.calls.append((float(video.sigma.detach().cpu().reshape(-1)[0]), tuple(video.latent.shape)))
        velocity = self.a * video.latent
        return video.latent - sigma_view * velocity, None


def parse_steps(value: str) -> list[int]:
    steps = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("expected at least one step count")
    if any(step < 2 for step in steps):
        raise argparse.ArgumentTypeError("all step counts must be >= 2")
    return steps


def fit_log_slope(ns: list[int], errors: list[float]) -> float:
    points = [(math.log(float(n)), math.log(float(e))) for n, e in zip(ns, errors) if e > 0.0 and math.isfinite(e)]
    if len(points) < 2:
        raise RuntimeError("Need at least two positive finite errors to fit a slope")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full production ContextFlowRFSolver2Sampler.sample() linear mock test.")
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("4,8,16,32,64,128"))
    parser.add_argument("--a", type=float, default=0.7)
    parser.add_argument("--dtype", choices=("float32", "float64", "bf16"), default="float64")
    parser.add_argument("--step-parameterization", choices=("velocity", "x0"), default="velocity")
    parser.add_argument("--step-compute-dtype", choices=("float32", "float64", "bf16"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition-strength", type=float, default=1.0)
    return parser.parse_args()


def make_video_tools(source: torch.Tensor) -> VideoLatentTools:
    return VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape.from_torch_shape(source.shape),
        fps=24.0,
    )


def run_once(
    *,
    source: torch.Tensor,
    steps: int,
    a: float,
    first_order: bool,
    condition_strength: float,
    step_parameterization: str,
    step_compute_dtype: torch.dtype,
) -> dict[str, float | int]:
    video_tools = make_video_tools(source)
    first_frame = source[:, :, :1].clone()
    conditionings = [
        VideoConditionByLatentIndex(
            latent=first_frame,
            strength=condition_strength,
            latent_idx=0,
        )
    ]
    condition = BranchCondition(
        context=torch.zeros((source.shape[0], 1, 4), device=source.device, dtype=source.dtype),
        conditionings=conditionings,
    )

    expected_state = video_tools.create_initial_state(source.device, source.dtype, source)
    expected_state = state_with_conditionings(expected_state, conditionings, video_tools)
    expected_state = video_tools.unpatchify(video_tools.clear_conditioning(expected_state))
    expected = expected_state.latent

    timesteps = torch.linspace(1.0, CONTEXTFLOW_SIGMA_MIN, steps + 1, device=source.device, dtype=source.dtype)
    model = LinearX0Model(a=a)
    sampler = ContextFlowRFSolver2Sampler(
        video_tools=video_tools,
        use_reverse_pullback=False,
        post_step_clamp=False,
        first_order=first_order,
        step_parameterization=step_parameterization,
        step_compute_dtype=step_compute_dtype,
    )
    result = sampler.sample(
        transformer=model,
        source_latent=source,
        condition=condition,
        timesteps=timesteps,
    )

    full_diff = result.reconstructed_latent - expected
    cond_diff = result.reconstructed_latent[:, :, :1] - expected[:, :, :1]
    free_diff = result.reconstructed_latent[:, :, 1:] - expected[:, :, 1:]
    inverted_expected = expected * math.exp(a)
    inverted_diff = result.inverted_latent - inverted_expected

    return {
        "calls": len(model.calls),
        "max_full": full_diff.abs().max().item(),
        "rms_full": full_diff.square().mean().sqrt().item(),
        "max_conditioned_frame": cond_diff.abs().max().item(),
        "max_free_frames": free_diff.abs().max().item(),
        "max_inverted_vs_exact": inverted_diff.abs().max().item(),
    }


def main() -> None:
    args = parse_args()
    dtype = {"float32": torch.float32, "float64": torch.float64, "bf16": torch.bfloat16}[args.dtype]
    step_compute_dtype = {"float32": torch.float32, "float64": torch.float64, "bf16": torch.bfloat16}[
        args.step_compute_dtype or args.dtype
    ]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    latent_shape = VideoLatentShape(batch=1, channels=3, frames=3, height=2, width=2)
    source = torch.randn(latent_shape.to_torch_shape(), device=device, dtype=dtype)

    rows = []
    for step_count in args.steps:
        row = run_once(
            source=source,
            steps=step_count,
            a=args.a,
            first_order=args.first_order,
            condition_strength=args.condition_strength,
            step_parameterization=args.step_parameterization,
            step_compute_dtype=step_compute_dtype,
        )
        rows.append((step_count, row))

    max_errors = [float(row["max_full"]) for _n, row in rows]
    rms_errors = [float(row["rms_full"]) for _n, row in rows]
    inv_errors = [float(row["max_inverted_vs_exact"]) for _n, row in rows]
    ns = [n for n, _row in rows]

    method = "first-order Euler" if args.first_order else "second-order midpoint/RF-Solver-2"
    print(f"field: v(x,sigma) = {args.a} * x")
    print(f"method: {method}")
    print(f"step_parameterization: {args.step_parameterization}")
    print(f"state_dtype: {args.dtype}")
    print(f"step_compute_dtype: {args.step_compute_dtype or args.dtype}")
    print(f"condition: VideoConditionByLatentIndex(latent=source first latent frame, strength={args.condition_strength}, latent_idx=0)")
    print("N,calls,max_full,rms_full,max_conditioned_frame,max_free_frames,max_inverted_vs_exact")
    for n, row in rows:
        print(
            f"{n},{row['calls']},{row['max_full']:.12g},{row['rms_full']:.12g},"
            f"{row['max_conditioned_frame']:.12g},{row['max_free_frames']:.12g},{row['max_inverted_vs_exact']:.12g}"
        )
    print(f"fit_log_max_full_error_vs_log_N_slope: {fit_log_slope(ns, max_errors):.6f}")
    print(f"fit_log_rms_full_error_vs_log_N_slope: {fit_log_slope(ns, rms_errors):.6f}")
    print(f"fit_log_inverted_exact_error_vs_log_N_slope: {fit_log_slope(ns, inv_errors):.6f}")


if __name__ == "__main__":
    main()
