"""Edge pathway attribution (Def 9.8).

For attention head edges, determines which input pathway (Q, K, V) carries
the causal signal by measuring activation delta norms at each projection.

    w_Q = ||dA_Q||_F / (||dA_Q||_F + ||dA_K||_F + ||dA_V||_F)
    w_K = ||dA_K||_F / (||dA_Q||_F + ||dA_K||_F + ||dA_V||_F)
    w_V = ||dA_V||_F / (||dA_Q||_F + ||dA_K||_F + ||dA_V||_F)

For non-attention targets, pathway is {residual: 1.0}.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class PathwayAttribution:
    """Attribution of an edge's causal pathway through Q/K/V projections."""

    source: str
    target: str
    w_Q: float = 0.0
    w_K: float = 0.0
    w_V: float = 0.0
    dominant_pathway: str = "residual"
    is_attention_target: bool = False

    @property
    def via(self) -> Dict[str, float]:
        if self.is_attention_target:
            return {"Q": self.w_Q, "K": self.w_K, "V": self.w_V}
        return {"residual": 1.0}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "via": self.via,
            "dominant_pathway": self.dominant_pathway,
        }


def compute_pathway_attribution(
    delta_A_Q: Tensor,
    delta_A_K: Tensor,
    delta_A_V: Tensor,
    source: str,
    target: str,
) -> PathwayAttribution:
    """Compute pathway weights from activation deltas at Q, K, V hooks.

    Each delta is the Frobenius norm of the activation change at that
    projection when the source component is ablated.
    """
    norm_Q = torch.norm(delta_A_Q).item()
    norm_K = torch.norm(delta_A_K).item()
    norm_V = torch.norm(delta_A_V).item()
    total = norm_Q + norm_K + norm_V

    if total < 1e-10:
        return PathwayAttribution(source=source, target=target)

    w_Q = norm_Q / total
    w_K = norm_K / total
    w_V = norm_V / total

    norms = {"Q": w_Q, "K": w_K, "V": w_V}
    dominant = max(norms, key=norms.get)

    return PathwayAttribution(
        source=source,
        target=target,
        w_Q=w_Q,
        w_K=w_K,
        w_V=w_V,
        dominant_pathway=dominant,
        is_attention_target=True,
    )


def compute_pathway_from_weight_deltas(
    delta_W_Q: Tensor,
    delta_W_K: Tensor,
    delta_W_V: Tensor,
    source: str,
    target: str,
) -> PathwayAttribution:
    """Alternative: compute pathway from weight matrix deltas (Frobenius norms)."""
    norm_Q = torch.norm(delta_W_Q).item()
    norm_K = torch.norm(delta_W_K).item()
    norm_V = torch.norm(delta_W_V).item()
    total = norm_Q + norm_K + norm_V

    if total < 1e-10:
        return PathwayAttribution(source=source, target=target)

    w_Q = norm_Q / total
    w_K = norm_K / total
    w_V = norm_V / total

    norms = {"Q": w_Q, "K": w_K, "V": w_V}
    dominant = max(norms, key=norms.get)

    return PathwayAttribution(
        source=source,
        target=target,
        w_Q=w_Q,
        w_K=w_K,
        w_V=w_V,
        dominant_pathway=dominant,
        is_attention_target=True,
    )


__all__ = [
    "PathwayAttribution",
    "compute_pathway_attribution",
    "compute_pathway_from_weight_deltas",
]
