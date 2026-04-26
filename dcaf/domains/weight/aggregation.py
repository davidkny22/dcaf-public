"""
Projection-level cluster delta aggregation.

Def 3.3 (§3 Multi-Path Discovery, Aggregated Cluster Deltas):
  δ̄_+^(proj) = Σ_{i∈T+} eff(i)·ΔW_i^(proj) / Σ_{i∈T+} eff(i)

Effectiveness-weighted average of delta MATRICES per cluster, using eff(i)^1
(linear power) for accurate direction estimates (see Effectiveness Power remark).
"""

from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from dcaf.core.structures import TrainingSignal


def compute_cluster_delta_matrix(
    proj: str,
    proj_deltas_by_signal: Dict[str, Dict[str, Tensor]],
    effectiveness: Dict[str, float],
    cluster_signals: List[str],
) -> Tensor:
    """
    Compute effectiveness-weighted average delta matrix for one cluster.

    δ̄^(proj) = Σ_{i∈cluster} eff(i)·ΔW_i^(proj) / Σ_{i∈cluster} eff(i)

    Args:
        proj: Projection ID
        proj_deltas_by_signal: {signal_id: {proj_id: delta_tensor}}
        effectiveness: {signal_id: eff_value}
        cluster_signals: List of signal IDs in this cluster

    Returns:
        Aggregated delta matrix (same shape as input projections)
    """
    numerator = None
    denominator = 0.0

    for sig_id in cluster_signals:
        eff = effectiveness.get(sig_id, 1.0)
        if eff <= 0:
            continue

        delta = proj_deltas_by_signal[sig_id][proj]

        if numerator is None:
            numerator = eff * delta
        else:
            numerator = numerator + eff * delta

        denominator += eff

    if numerator is None or denominator == 0:
        # Return zeros with shape of any available delta
        for sig_id in cluster_signals:
            if sig_id in proj_deltas_by_signal and proj in proj_deltas_by_signal[sig_id]:
                return torch.zeros_like(proj_deltas_by_signal[sig_id][proj])
        raise ValueError(f"No deltas found for projection {proj}")

    return numerator / denominator


def compute_cluster_deltas(
    proj: str,
    proj_deltas_by_signal: Dict[str, Dict[str, Tensor]],
    effectiveness: Dict[str, float],
    plus_signals: List[str],
    minus_signals: List[str],
) -> Tuple[Tensor, Tensor]:
    """
    Compute aggregated T+ and T- delta matrices for a projection.

    Args:
        proj: Projection ID
        proj_deltas_by_signal: {signal_id: {proj_id: delta_tensor}}
        effectiveness: {signal_id: eff_value}
        plus_signals: Signal IDs in T+ cluster
        minus_signals: Signal IDs in T- cluster

    Returns:
        (delta_bar_plus, delta_bar_minus) tuple of matrices
    """
    delta_bar_plus = compute_cluster_delta_matrix(
        proj, proj_deltas_by_signal, effectiveness, plus_signals,
    )
    delta_bar_minus = compute_cluster_delta_matrix(
        proj, proj_deltas_by_signal, effectiveness, minus_signals,
    )
    return delta_bar_plus, delta_bar_minus


__all__ = [
    "compute_cluster_delta_matrix",
    "compute_cluster_deltas",
]
