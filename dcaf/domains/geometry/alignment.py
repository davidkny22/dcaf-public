"""
Alignment matrix and cluster metrics.

Def 6.8-6.10 (§6.4):
M^(k)[i, j] = cos(d_i^(k), d_j^(k))
coh+, coh-, opp, orth metrics
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class ClusterMetrics:
    """
    Cluster metrics for a component.

    Attributes:
        coh_plus: Target cluster coherence
        coh_minus: Opposite cluster coherence
        opposition: Cross-cluster opposition
        orthogonality: Baseline orthogonality
    """
    coh_plus: float
    coh_minus: float
    opposition: float
    orthogonality: float


def compute_alignment_matrix(
    directions: Dict[str, Tensor],
) -> Tensor:
    """
    Compute alignment matrix of cosine similarities.

    M^(k)[i, j] = cos(d_i^(k), d_j^(k))

    Args:
        directions: {signal_id: direction} where direction is [d]

    Returns:
        Alignment matrix [n_signals, n_signals]
    """
    signal_ids = list(directions.keys())
    n = len(signal_ids)

    if n == 0:
        return torch.zeros(0, 0)

    M = torch.zeros(n, n)

    for i, sig_i in enumerate(signal_ids):
        for j, sig_j in enumerate(signal_ids):
            M[i, j] = F.cosine_similarity(
                directions[sig_i].unsqueeze(0),
                directions[sig_j].unsqueeze(0)
            ).item()

    return M


def compute_alignment_matrix_indexed(
    directions: Dict[str, Tensor],
) -> tuple:
    """
    Compute alignment matrix with index mapping.

    Args:
        directions: {signal_id: direction}

    Returns:
        (alignment_matrix, signal_id_to_index, index_to_signal_id)
    """
    signal_ids = list(directions.keys())
    sig_to_idx = {sig: i for i, sig in enumerate(signal_ids)}
    idx_to_sig = {i: sig for sig, i in sig_to_idx.items()}

    M = compute_alignment_matrix(directions)

    return M, sig_to_idx, idx_to_sig


def compute_cluster_coherence(
    M: Tensor,
    cluster_indices: List[int],
) -> float:
    """
    Compute within-cluster coherence.

    coh = (1/|C|²) · Σ_{i,j∈C} M[i, j]

    Args:
        M: Alignment matrix [n, n]
        cluster_indices: Indices of cluster members

    Returns:
        Coherence value in [0, 1]
    """
    if not cluster_indices:
        return 0.0

    n = len(cluster_indices)
    total = sum(
        M[i, j].item()
        for i in cluster_indices
        for j in cluster_indices
    )

    return total / (n * n)


def compute_cluster_opposition(
    M: Tensor,
    T_plus_indices: List[int],
    T_minus_indices: List[int],
) -> float:
    """
    Compute cross-cluster opposition.

    opp = -(1/(|T+||T-|)) · Σ_{i∈T+,j∈T-} M[i, j]

    Args:
        M: Alignment matrix
        T_plus_indices: Target cluster indices
        T_minus_indices: Opposite cluster indices

    Returns:
        Opposition value (higher = more opposed)
    """
    if not T_plus_indices or not T_minus_indices:
        return 0.0

    total = sum(
        M[i, j].item()
        for i in T_plus_indices
        for j in T_minus_indices
    )

    return -total / (len(T_plus_indices) * len(T_minus_indices))


def compute_baseline_orthogonality(
    M: Tensor,
    behavioral_indices: List[int],
    baseline_index: int,
) -> float:
    """
    Compute baseline orthogonality.

    orth = 1 - (1/|T±|) · Σ_{i∈T±} |M[i, t₀]|

    Args:
        M: Alignment matrix
        behavioral_indices: Indices of T+ ∪ T-
        baseline_index: Index of baseline signal t₀

    Returns:
        Orthogonality value (higher = more orthogonal to baseline)
    """
    if not behavioral_indices:
        return 1.0

    total_abs = sum(
        abs(M[i, baseline_index].item())
        for i in behavioral_indices
    )

    return 1.0 - total_abs / len(behavioral_indices)


def compute_cluster_metrics(
    M: Tensor,
    T_plus_indices: List[int],
    T_minus_indices: List[int],
    baseline_index: int,
) -> ClusterMetrics:
    """
    Compute all cluster metrics.

    Args:
        M: Alignment matrix
        T_plus_indices: Target cluster indices
        T_minus_indices: Opposite cluster indices
        baseline_index: Index of baseline signal

    Returns:
        ClusterMetrics with coh+, coh-, opp, orth
    """
    behavioral_indices = T_plus_indices + T_minus_indices

    return ClusterMetrics(
        coh_plus=compute_cluster_coherence(M, T_plus_indices),
        coh_minus=compute_cluster_coherence(M, T_minus_indices),
        opposition=compute_cluster_opposition(M, T_plus_indices, T_minus_indices),
        orthogonality=compute_baseline_orthogonality(M, behavioral_indices, baseline_index),
    )


def compute_cluster_metrics_from_directions(
    directions: Dict[str, Tensor],
    T_plus_signals: List[str],
    T_minus_signals: List[str],
    baseline_signal: str,
) -> ClusterMetrics:
    """
    Compute cluster metrics directly from directions.

    Args:
        directions: {signal_id: direction}
        T_plus_signals: Target cluster signal IDs
        T_minus_signals: Opposite cluster signal IDs
        baseline_signal: Baseline signal ID

    Returns:
        ClusterMetrics
    """
    M, sig_to_idx, _ = compute_alignment_matrix_indexed(directions)

    T_plus_indices = [sig_to_idx[s] for s in T_plus_signals if s in sig_to_idx]
    T_minus_indices = [sig_to_idx[s] for s in T_minus_signals if s in sig_to_idx]
    if baseline_signal not in sig_to_idx:
        raise ValueError(
            f"baseline_signal {baseline_signal!r} was not found in directions; "
            "cannot compute baseline orthogonality"
        )
    baseline_index = sig_to_idx[baseline_signal]

    return compute_cluster_metrics(M, T_plus_indices, T_minus_indices, baseline_index)


def get_alignment_summary(
    M: Tensor,
) -> Dict[str, Any]:
    """
    Summary statistics for alignment matrix.

    Args:
        M: Alignment matrix

    Returns:
        Summary dict with mean, std, min, max
    """
    if M.numel() == 0:
        return {"count": 0}

    # Exclude diagonal for off-diagonal stats
    n = M.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = M[mask]

    return {
        "n_signals": n,
        "mean_alignment": M.mean().item(),
        "std_alignment": M.std().item(),
        "min_alignment": M.min().item(),
        "max_alignment": M.max().item(),
        "off_diag_mean": off_diag.mean().item() if off_diag.numel() > 0 else 0.0,
        "off_diag_std": off_diag.std().item() if off_diag.numel() > 1 else 0.0,
    }


__all__ = [
    "ClusterMetrics",
    "compute_alignment_matrix",
    "compute_alignment_matrix_indexed",
    "compute_cluster_coherence",
    "compute_cluster_opposition",
    "compute_baseline_orthogonality",
    "compute_cluster_metrics",
    "compute_cluster_metrics_from_directions",
    "get_alignment_summary",
]
