"""
Generalization computation.

def:generalization:
gen(k) = (1/|T+|) · Σ_{i∈T+} (pred_OOD(d_i) / pred_within(d_i))
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from torch import Tensor

from dcaf.core.defaults import (
    EPS_GENERAL,
    GEN_THRESHOLD,
    GENERALIZATION_GAP_THRESHOLD,
    OVERFITTING_GAP_THRESHOLD,
)

from .predictivity import compute_predictivity


@dataclass
class GeneralizationResult:
    """
    Generalization result for a component.

    Attributes:
        gen: Generalization score (ratio of OOD to within-distribution)
        per_signal_ratio: {signal: pred_ood / pred_within} for each signal
        mean_pred_within: Mean within-distribution predictivity
        mean_pred_ood: Mean OOD predictivity
        gap: Generalization gap (pred_within - pred_ood)
    """
    gen: float
    per_signal_ratio: Dict[str, float]
    mean_pred_within: float
    mean_pred_ood: float
    gap: float


def compute_generalization_ratio(
    pred_within: float,
    pred_ood: float,
    eps: float = EPS_GENERAL,
) -> float:
    """
    Compute generalization ratio for a single direction.

    ratio = pred_OOD / pred_within

    Args:
        pred_within: Within-distribution predictivity
        pred_ood: Out-of-distribution predictivity
        eps: Numerical stability

    Returns:
        Generalization ratio (higher = better generalization)
    """
    if pred_within < eps:
        return 0.0

    return pred_ood / pred_within


def compute_generalization(
    directions: Dict[str, Tensor],
    A_within: Tensor,
    labels_within: Tensor,
    A_ood: Tensor,
    labels_ood: Tensor,
    T_plus_signals: List[str],
    eps: float = EPS_GENERAL,
) -> GeneralizationResult:
    """
    Compute generalization score for a component.

    gen(k) = (1/|T+|) · Σ_{i∈T+} (pred_OOD(d_i) / pred_within(d_i))

    Interpretation:
    - gen ≈ 1.0: Perfect generalization
    - gap < 0.1: Direction captures generalizable concept
    - gap > 0.2: Direction overfits to template artifacts
    - pred_OOD > 0.7: Strong evidence of real concept

    Args:
        directions: {signal_id: direction}
        A_within: Within-distribution activations
        labels_within: Within-distribution labels
        A_ood: OOD activations
        labels_ood: OOD labels
        T_plus_signals: Target cluster signal IDs
        eps: Numerical stability

    Returns:
        GeneralizationResult
    """
    ratios = {}
    pred_withins = []
    pred_oods = []

    for signal in T_plus_signals:
        if signal not in directions:
            continue

        direction = directions[signal]

        # Compute within-distribution predictivity
        result_within = compute_predictivity(direction, A_within, labels_within)
        pred_within = result_within.auc

        # Compute OOD predictivity
        result_ood = compute_predictivity(direction, A_ood, labels_ood)
        pred_ood = result_ood.auc

        # Compute ratio
        ratio = compute_generalization_ratio(pred_within, pred_ood, eps)
        ratios[signal] = ratio

        pred_withins.append(pred_within)
        pred_oods.append(pred_ood)

    if not ratios:
        return GeneralizationResult(
            gen=0.0,
            per_signal_ratio={},
            mean_pred_within=0.0,
            mean_pred_ood=0.0,
            gap=0.0,
        )

    # Compute aggregates
    mean_ratio = sum(ratios.values()) / len(ratios)
    mean_pred_within = sum(pred_withins) / len(pred_withins)
    mean_pred_ood = sum(pred_oods) / len(pred_oods)
    gap = mean_pred_within - mean_pred_ood

    return GeneralizationResult(
        gen=mean_ratio,
        per_signal_ratio=ratios,
        mean_pred_within=mean_pred_within,
        mean_pred_ood=mean_pred_ood,
        gap=gap,
    )


def compute_generalization_simple(
    direction: Tensor,
    A_within: Tensor,
    labels_within: Tensor,
    A_ood: Tensor,
    labels_ood: Tensor,
    eps: float = EPS_GENERAL,
) -> Tuple[float, float]:
    """
    Compute generalization for a single direction.

    Args:
        direction: Contrastive direction
        A_within: Within-distribution activations
        labels_within: Within-distribution labels
        A_ood: OOD activations
        labels_ood: OOD labels
        eps: Numerical stability

    Returns:
        (generalization_ratio, gap)
    """
    pred_within = compute_predictivity(direction, A_within, labels_within).auc
    pred_ood = compute_predictivity(direction, A_ood, labels_ood).auc

    ratio = compute_generalization_ratio(pred_within, pred_ood, eps)
    gap = pred_within - pred_ood

    return ratio, gap


def is_generalizable(
    gen: float,
    gap: float,
    gen_threshold: float = GEN_THRESHOLD,
    gap_threshold: float = GENERALIZATION_GAP_THRESHOLD,
) -> bool:
    """
    Check if direction captures generalizable concept.

    Args:
        gen: Generalization score
        gap: Generalization gap
        gen_threshold: Minimum acceptable OOD/within predictivity ratio
        gap_threshold: Maximum acceptable gap

    Returns:
        True if direction generalizes well
    """
    return gen >= gen_threshold and gap < gap_threshold


def is_overfitting(
    gap: float,
    gap_threshold: float = OVERFITTING_GAP_THRESHOLD,
) -> bool:
    """
    Check if direction overfits to template artifacts.

    Args:
        gap: Generalization gap
        gap_threshold: Overfitting threshold

    Returns:
        True if likely overfitting
    """
    return gap > gap_threshold


def get_generalization_summary(
    result: GeneralizationResult,
) -> Dict[str, Any]:
    """
    Summary statistics for generalization result.

    Args:
        result: GeneralizationResult

    Returns:
        Summary dict
    """
    return {
        "gen": result.gen,
        "gap": result.gap,
        "mean_pred_within": result.mean_pred_within,
        "mean_pred_ood": result.mean_pred_ood,
        "n_signals": len(result.per_signal_ratio),
        "is_generalizable": is_generalizable(result.gen, result.gap),
        "is_overfitting": is_overfitting(result.gap),
    }


__all__ = [
    "GeneralizationResult",
    "compute_generalization_ratio",
    "compute_generalization",
    "compute_generalization_simple",
    "is_generalizable",
    "is_overfitting",
    "get_generalization_summary",
]
