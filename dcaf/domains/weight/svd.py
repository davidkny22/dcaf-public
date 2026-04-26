"""
SVD diagnostics for projection weight deltas.

Def 4.5 (§4 Weight Analysis, SVD Decomposition of Cluster Deltas):
  Spectral decomposition of aggregated cluster deltas reveals:
  - rank1(proj): how concentrated the weight change is (rank-1 fraction)
  - spec_opp(proj): whether T+ and T- oppose along the primary axis of change

These diagnostics are optional metadata and do not enter the confidence formula.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from dcaf.core.structures import SVDDiagnostics


def compute_svd_diagnostics(
    delta_bar_plus: Tensor,
    delta_bar_minus: Tensor,
) -> SVDDiagnostics:
    """
    Compute spectral decomposition diagnostics for a projection.

    Args:
        delta_bar_plus: Aggregated T+ delta matrix
        delta_bar_minus: Aggregated T- delta matrix

    Returns:
        SVDDiagnostics with rank-1 fraction, singular values,
        and spectral opposition
    """
    # SVD of T+ aggregated delta
    U_p, S_p, Vh_p = torch.linalg.svd(delta_bar_plus.float(), full_matrices=False)
    U_m, S_m, Vh_m = torch.linalg.svd(delta_bar_minus.float(), full_matrices=False)

    # Rank-1 fraction: how much variance is captured by top component
    total_var = torch.sum(S_p ** 2).item()
    rank_1_fraction = (S_p[0] ** 2 / total_var).item() if total_var > 0 else 0.0

    # Top singular values
    top_3 = [S_p[i].item() for i in range(min(3, len(S_p)))]

    # Spectral opposition: do T+ and T- primary directions oppose?
    if len(Vh_p) > 0 and len(Vh_m) > 0:
        spectral_opp = -F.cosine_similarity(
            Vh_p[0].unsqueeze(0), Vh_m[0].unsqueeze(0),
        ).item()
    else:
        spectral_opp = 0.0

    return SVDDiagnostics(
        rank_1_fraction=rank_1_fraction,
        top_singular_value=S_p[0].item() if len(S_p) > 0 else 0.0,
        top_3_singular_values=top_3,
        spectral_opposition=spectral_opp,
    )


__all__ = ["compute_svd_diagnostics"]
