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
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.inversion.config import BranchCondition
from ltx_pipelines.inversion.rf_samplers import ContextFlowRFSolver2Sampler
from ltx_pipelines.inversion.schedule import CONTEXTFLOW_SIGMA_MIN


class LinearX0Model:
    """Fake LTX X0Model for the analytic field v(x, sigma) = a * x.

    The LTX-style x0 prediction that corresponds to a velocity field v is:
    D(x, sigma) = x - sigma * v(x, sigma).
    For this test, D(x, sigma) = x - sigma * a * x.
    """

    def __init__(self, a: float) -> None:
        self.a = float(a)
        self.calls = 0

    def __call__(self, video, audio, perturbations):
        del audio, perturbations
        sigma_view = video.timesteps.to(device=video.latent.device, dtype=video.latent.dtype)
        while sigma_view.dim() < video.latent.dim():
            sigma_view = sigma_view.unsqueeze(-1)
        self.calls += 1
        velocity = self.a * video.latent
        return video.latent - sigma_view * velocity, None


def parse_steps(value: str) -> list[int]:
    steps = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("expected at least one step count")
    if any(step < 2 for step in steps):
        raise argparse.ArgumentTypeError("all step counts must be >= 2")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convergence test for ContextFlowRFSolver2Sampler on v(x,sigma)=a*x.")
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("4,8,16,32,64,128"))
    parser.add_argument("--a", type=float, default=0.7, help="Scalar coefficient in v(x, sigma)=a*x.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--first-order", action="store_true", help="Use first-order Euler updates instead of midpoint/RF-Solver-2.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--atol", type=float, default=None, help="Optional maximum tolerated error at the finest step count.")
    return parser.parse_args()


def fit_log_slope(ns: list[int], errors: list[float]) -> tuple[float, float]:
    points = [(math.log(float(n)), math.log(float(e))) for n, e in zip(ns, errors) if e > 0.0 and math.isfinite(e)]
    if len(points) < 2:
        raise RuntimeError("Need at least two positive finite errors to fit a slope")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def run_once(*, steps: int, a: float, dtype: torch.dtype, device: torch.device, source: torch.Tensor, first_order: bool) -> tuple[float, float, int]:
    latent_shape = VideoLatentShape.from_torch_shape(source.shape)
    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=latent_shape,
        fps=24.0,
    )
    condition = BranchCondition(
        context=torch.zeros((source.shape[0], 1, 4), device=device, dtype=dtype),
        conditionings=[],
    )
    timesteps = torch.linspace(1.0, CONTEXTFLOW_SIGMA_MIN, steps + 1, device=device, dtype=dtype)
    model = LinearX0Model(a=a)
    sampler = ContextFlowRFSolver2Sampler(
        video_tools=video_tools,
        use_reverse_pullback=False,
        post_step_clamp=False,
        first_order=first_order,
    )
    result = sampler.sample(
        transformer=model,
        source_latent=source,
        condition=condition,
        timesteps=timesteps,
    )
    diff = result.reconstructed_latent - source
    return diff.abs().max().item(), diff.square().mean().sqrt().item(), model.calls


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Keep the shape tiny; this is a solver test, not a model/VAE/data test.
    latent_shape = VideoLatentShape(batch=1, channels=3, frames=2, height=2, width=2)
    source = torch.randn(latent_shape.to_torch_shape(), device=device, dtype=dtype)

    ns: list[int] = []
    max_errors: list[float] = []
    rms_errors: list[float] = []
    calls: list[int] = []
    for step_count in args.steps:
        max_error, rms_error, call_count = run_once(
            steps=step_count,
            a=args.a,
            dtype=dtype,
            device=device,
            source=source,
            first_order=args.first_order,
        )
        ns.append(step_count)
        max_errors.append(max_error)
        rms_errors.append(rms_error)
        calls.append(call_count)

    max_slope, max_intercept = fit_log_slope(ns, max_errors)
    rms_slope, rms_intercept = fit_log_slope(ns, rms_errors)
    method = "first-order Euler" if args.first_order else "second-order midpoint/RF-Solver-2"

    print(f"field: v(x,sigma) = {args.a} * x")
    print(f"method: {method}")
    print(f"dtype_requested: {args.dtype}")
    print("N,calls,max_abs_error,rms_error")
    for n, call_count, max_error, rms_error in zip(ns, calls, max_errors, rms_errors):
        print(f"{n},{call_count},{max_error:.12g},{rms_error:.12g}")
    print(f"fit_log_max_abs_error_vs_log_N_slope: {max_slope:.6f}")
    print(f"fit_log_rms_error_vs_log_N_slope: {rms_slope:.6f}")
    print(f"estimated_max_abs_order: {-max_slope:.6f}")
    print(f"estimated_rms_order: {-rms_slope:.6f}")
    print(f"fit_max_abs_intercept: {max_intercept:.6f}")
    print(f"fit_rms_intercept: {rms_intercept:.6f}")

    if args.atol is not None and max_errors[-1] > args.atol:
        raise SystemExit(f"FAIL: finest max error {max_errors[-1]:.12g} > atol {args.atol:.12g}")


if __name__ == "__main__":
    main()
