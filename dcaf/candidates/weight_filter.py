"""
Weight domain filtering for candidate selection.

Implements: H_W = {p : C_W⁽ᵖ⁾ ≥ τ_W}

Weight filtering is the discovery phase - it identifies parameter-level
candidates based purely on how they responded to training signals.
"""

from typing import Any, Dict, List, Set, Tuple

from dcaf.core.defaults import PERCENTILE_FILTER, TAU_W_DEFAULT, TOP_K_CANDIDATES


def filter_by_weight_confidence(
    weight_confidences: Dict[Any, float],
    tau_W: float = TAU_W_DEFAULT,
) -> Set[Any]:
    """
    Filter parameters by weight confidence threshold.

    H_W = {p : C_W⁽ᵖ⁾ ≥ τ_W}

    Args:
        weight_confidences: {param_id: C_W} mapping
        tau_W: Weight confidence threshold

    Returns:
        Set of param_ids that pass threshold
    """
    return {
        param_id for param_id, c_w in weight_confidences.items()
        if c_w >= tau_W
    }


def filter_by_weight_percentile(
    weight_confidences: Dict[Any, float],
    percentile: float = PERCENTILE_FILTER,
) -> Set[Any]:
    """
    Filter parameters to top (100 - percentile)%.

    Args:
        weight_confidences: {param_id: C_W} mapping
        percentile: Percentile threshold (e.g., 85 = top 15%)

    Returns:
        Set of param_ids in top percentile
    """
    if not weight_confidences:
        return set()

    sorted_items = sorted(
        weight_confidences.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    cutoff_idx = int(len(sorted_items) * (100 - percentile) / 100)
    cutoff_idx = max(1, cutoff_idx)  # At least 1

    return {param_id for param_id, _ in sorted_items[:cutoff_idx]}


def filter_by_weight_top_k(
    weight_confidences: Dict[Any, float],
    k: int = TOP_K_CANDIDATES,
) -> Set[Any]:
    """
    Filter to top-k parameters by weight confidence.

    Args:
        weight_confidences: {param_id: C_W} mapping
        k: Number of top candidates to keep

    Returns:
        Set of top-k param_ids
    """
    if not weight_confidences:
        return set()

    sorted_items = sorted(
        weight_confidences.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {param_id for param_id, _ in sorted_items[:k]}


def rank_by_weight_confidence(
    weight_confidences: Dict[Any, float],
) -> List[Tuple[Any, float]]:
    """
    Rank parameters by weight confidence.

    Args:
        weight_confidences: {param_id: C_W} mapping

    Returns:
        [(param_id, C_W), ...] sorted by C_W descending
    """
    return sorted(
        weight_confidences.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def compute_weight_statistics(
    weight_confidences: Dict[Any, float],
) -> Dict[str, float]:
    """
    Compute statistics on weight confidence distribution.

    Args:
        weight_confidences: {param_id: C_W} mapping

    Returns:
        Statistics dict
    """
    if not weight_confidences:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    values = list(weight_confidences.values())
    sorted_vals = sorted(values)
    n = len(values)

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "median": (sorted_vals[n // 2] + sorted_vals[(n - 1) // 2]) / 2,
    }


__all__ = [
    "filter_by_weight_confidence",
    "filter_by_weight_percentile",
    "filter_by_weight_top_k",
    "rank_by_weight_confidence",
    "compute_weight_statistics",
]
