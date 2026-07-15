from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ltx_pipelines.contextflow.instrumentation.memory import tensor_nbytes
from ltx_pipelines.contextflow.instrumentation.tensor_stats import mean_cosine_similarity, tensor_stats


@dataclass
class ContextFlowInstrumentation:
    records: list[dict[str, float | int | str]] = field(default_factory=list)

    def record_attention(
        self,
        *,
        step_index: int,
        num_steps: int,
        layer_index: int,
        target_q: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        augmented_k: torch.Tensor,
        augmented_v: torch.Tensor,
    ) -> None:
        self.records.append(
            {
                "step_index": step_index,
                "step_progress": step_index / float(max(num_steps - 1, 1)),
                "layer_index": layer_index,
                "target_q_tokens": int(target_q.shape[1]),
                "target_k_tokens": int(target_k.shape[1]),
                "target_v_tokens": int(target_v.shape[1]),
                "source_k_tokens": int(source_k.shape[1]),
                "source_v_tokens": int(source_v.shape[1]),
                "augmented_k_tokens": int(augmented_k.shape[1]),
                "augmented_v_tokens": int(augmented_v.shape[1]),
                "dtype": str(target_q.dtype),
                "device": str(target_q.device),
                "source_cache_bytes": tensor_nbytes(source_k) + tensor_nbytes(source_v),
                "target_q_norm": tensor_stats(target_q)["norm"],
                "source_target_k_cosine": mean_cosine_similarity(target_k, source_k),
                "source_target_v_cosine": mean_cosine_similarity(target_v, source_v),
            }
        )
