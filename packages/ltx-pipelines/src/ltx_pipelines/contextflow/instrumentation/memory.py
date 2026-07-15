from __future__ import annotations

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()
