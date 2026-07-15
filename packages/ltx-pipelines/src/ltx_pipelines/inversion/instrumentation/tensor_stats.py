from __future__ import annotations

import torch


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    sample = tensor.detach().float()
    return {
        "mean": float(sample.mean().item()),
        "std": float(sample.std().item()),
        "norm": float(sample.norm().item()),
    }


def mean_cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_flat = lhs.detach().float().reshape(-1, lhs.shape[-1])
    rhs_flat = rhs.detach().float().reshape(-1, rhs.shape[-1])
    lhs_norm = torch.nn.functional.normalize(lhs_flat, dim=-1)
    rhs_norm = torch.nn.functional.normalize(rhs_flat, dim=-1)
    return float((lhs_norm * rhs_norm).sum(dim=-1).mean().item())
