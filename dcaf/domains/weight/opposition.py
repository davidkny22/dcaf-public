"""
Projection-level opposition degree via matrix cosine similarity.

def:opposition-degree; def:opposition-degree-full-methodology:
  cos_opp = cos(flatten(δ̄_+^(proj)), flatten(δ̄_-^(proj)))
  opp_degree = max(0, -cos_opp)

Self-normalizing — no RMS normalization needed.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from dcaf.core.defaults import TAU_OPP


def compute_opposition_degree(
    delta_bar_plus: Tensor,
    delta_bar_minus: Tensor,
) -> Tuple[float, float]:
    """
    Compute opposition degree between T+ and T- cluster deltas.

    cos_opp = cos(flatten(δ̄_+), flatten(δ̄_-))
    opp_degree = max(0, -cos_opp)

    When T+ and T- push the projection in opposite directions:
      cos_opp ≈ -1, opp_degree ≈ 1
    When they push in the same direction:
      cos_opp ≈ +1, opp_degree = 0

    Args:
        delta_bar_plus: Aggregated T+ delta matrix
        delta_bar_minus: Aggregated T- delta matrix

    Returns:
        (cos_opposition, opp_degree) tuple
    """
    flat_plus = delta_bar_plus.flatten().float()
    flat_minus = delta_bar_minus.flatten().float()

    norm_plus = torch.norm(flat_plus)
    norm_minus = torch.norm(flat_minus)

    if norm_plus < 1e-8 or norm_minus < 1e-8:
        return 0.0, 0.0

    cos_opp = F.cosine_similarity(
        flat_plus.unsqueeze(0), flat_minus.unsqueeze(0),
    ).item()
    opp_degree = max(0.0, -cos_opp)

    return cos_opp, opp_degree


def is_bidirectional(opp_degree: float, tau_opp: float = TAU_OPP) -> bool:
    """Check if a projection shows bidirectional control."""
    return opp_degree > tau_opp


__all__ = [
    "compute_opposition_degree",
    "is_bidirectional",
]
