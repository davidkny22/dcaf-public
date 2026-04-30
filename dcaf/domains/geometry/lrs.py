"""
Linear Representation Score computation.

Implements Def 6.14 (§6.6):
LRS(k) = [(Σ_{i=1}^6 w_i · (x_i + ε)^p) / (Σ_{i=1}^6 w_i)]^{1/p}
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math

from dcaf.core.defaults import P_LRS, EPSILON_TRI


def _clamp_unit(value: float) -> float:
    """Clamp a scalar metric into the unit interval before confidence aggregation."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class LRSBreakdown:
    """
    LRS component breakdown.

    The six components:
    1. coh_plus: Target cluster coherence
    2. coh_minus: Opposite cluster coherence
    3. opposition: Cluster opposition
    4. orthogonality: Baseline orthogonality
    5. confound_independence: Confound independence ρ_c
    6. predictivity_gain_norm: Normalized predictivity gain (1 + Δ_pred)/2

    Attributes:
        coh_plus: x_1
        coh_minus: x_2
        opposition: x_3
        orthogonality: x_4
        confound_independence: x_5
        predictivity_gain_norm: x_6
        weights: [w_1, ..., w_6]
    """
    coh_plus: float
    coh_minus: float
    opposition: float
    orthogonality: float
    confound_independence: float
    predictivity_gain_norm: float
    weights: List[float]


@dataclass
class LRSResult:
    """
    LRS computation result.

    Attributes:
        lrs: Final LRS score
        breakdown: Component breakdown
        p: Power mean parameter used
        epsilon: Smoothing parameter used
    """
    lrs: float
    breakdown: LRSBreakdown
    p: float
    epsilon: float


def power_mean(
    values: List[float],
    weights: List[float],
    p: float = P_LRS,
    epsilon: float = EPSILON_TRI,
) -> float:
    """
    Compute weighted power mean.

    M_p = [(Σ w_i · (x_i + ε)^p) / (Σ w_i)]^{1/p}

    Special cases:
    - p → 0: Geometric mean
    - p = 1: Arithmetic mean
    - p = -1: Harmonic mean

    Args:
        values: [x_1, ..., x_n]
        weights: [w_1, ..., w_n]
        p: Power parameter (default 0.5)
        epsilon: Smoothing to prevent zeros

    Returns:
        Power mean value
    """
    if not values or not weights:
        return 0.0

    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    # Metrics may be signed upstream; power means with fractional p require
    # nonnegative inputs. Clamp before smoothing to preserve a valid score.
    smoothed = [_clamp_unit(x) + epsilon for x in values]

    # Compute weighted power sum
    numerator = sum(w * (x ** p) for w, x in zip(weights, smoothed))
    denominator = sum(weights)

    if denominator == 0:
        return 0.0

    if p == 0:
        log_sum = sum(w * math.log(x) for w, x in zip(weights, smoothed))
        return math.exp(log_sum / denominator)

    return (numerator / denominator) ** (1.0 / p)


def compute_lrs(
    coh_plus: float,
    coh_minus: float,
    opposition: float,
    orthogonality: float,
    confound_independence: float,
    predictivity_gain: float,
    p: float = P_LRS,
    epsilon: float = EPSILON_TRI,
    weights: Optional[List[float]] = None,
) -> LRSResult:
    """
    Compute Linear Representation Score.

    LRS(k) = [(Σ w_i · (x_i + ε)^p) / (Σ w_i)]^{1/p}

    Interpretation:
    - LRS > 0.7: Strong linear representation
    - LRS ∈ [0.4, 0.7]: Moderate — some noise or non-linearity
    - LRS < 0.4: Weak — behavior may be gated or distributed

    Args:
        coh_plus: Target cluster coherence (x_1)
        coh_minus: Opposite cluster coherence (x_2)
        opposition: Cluster opposition (x_3)
        orthogonality: Baseline orthogonality (x_4)
        confound_independence: Confound independence (x_5)
        predictivity_gain: Raw predictivity gain (will be normalized)
        p: Power mean parameter
        epsilon: Smoothing parameter
        weights: Optional custom weights (default uniform)

    Returns:
        LRSResult with score and breakdown
    """
    # Normalize predictivity gain: (1 + Δ_pred) / 2
    predictivity_gain_norm = _clamp_unit((1.0 + predictivity_gain) / 2.0)

    # Default uniform weights
    if weights is None:
        weights = [1.0] * 6

    # Build component list
    values = [
        _clamp_unit(coh_plus),
        _clamp_unit(coh_minus),
        _clamp_unit(opposition),
        _clamp_unit(orthogonality),
        _clamp_unit(confound_independence),
        predictivity_gain_norm,
    ]

    # Compute power mean
    lrs = power_mean(values, weights, p, epsilon)

    breakdown = LRSBreakdown(
        coh_plus=values[0],
        coh_minus=values[1],
        opposition=values[2],
        orthogonality=values[3],
        confound_independence=values[4],
        predictivity_gain_norm=predictivity_gain_norm,
        weights=weights,
    )

    return LRSResult(
        lrs=lrs,
        breakdown=breakdown,
        p=p,
        epsilon=epsilon,
    )


def compute_lrs_from_breakdown(
    breakdown: LRSBreakdown,
    p: float = P_LRS,
    epsilon: float = EPSILON_TRI,
) -> float:
    """
    Compute LRS from pre-computed breakdown.

    Args:
        breakdown: LRSBreakdown with all components
        p: Power mean parameter
        epsilon: Smoothing parameter

    Returns:
        LRS score
    """
    values = [
        breakdown.coh_plus,
        breakdown.coh_minus,
        breakdown.opposition,
        breakdown.orthogonality,
        breakdown.confound_independence,
        breakdown.predictivity_gain_norm,
    ]

    return power_mean(values, breakdown.weights, p, epsilon)


def compute_lrs_batch(
    components_data: Dict[str, dict],
    p: float = P_LRS,
    epsilon: float = EPSILON_TRI,
) -> Dict[str, LRSResult]:
    """
    Compute LRS for multiple components.

    Args:
        components_data: {component_id: {
            "coh_plus": float,
            "coh_minus": float,
            "opposition": float,
            "orthogonality": float,
            "confound_independence": float,
            "predictivity_gain": float,
        }}
        p: Power mean parameter
        epsilon: Smoothing parameter

    Returns:
        {component_id: LRSResult}
    """
    return {
        component: compute_lrs(
            coh_plus=data["coh_plus"],
            coh_minus=data["coh_minus"],
            opposition=data["opposition"],
            orthogonality=data["orthogonality"],
            confound_independence=data["confound_independence"],
            predictivity_gain=data["predictivity_gain"],
            p=p,
            epsilon=epsilon,
        )
        for component, data in components_data.items()
    }


def is_strong_representation(
    lrs: float,
    threshold: float = 0.7,
) -> bool:
    """
    Check if LRS indicates strong linear representation.

    Args:
        lrs: LRS score
        threshold: Strong representation threshold

    Returns:
        True if strong linear representation
    """
    return lrs > threshold


def is_weak_representation(
    lrs: float,
    threshold: float = 0.4,
) -> bool:
    """
    Check if LRS indicates weak representation.

    Args:
        lrs: LRS score
        threshold: Weak representation threshold

    Returns:
        True if weak (behavior may be gated or distributed)
    """
    return lrs < threshold


def get_lrs_summary(
    result: LRSResult,
) -> Dict[str, Any]:
    """
    Summary of LRS result.

    Args:
        result: LRSResult

    Returns:
        Summary dict
    """
    return {
        "lrs": result.lrs,
        "is_strong": is_strong_representation(result.lrs),
        "is_weak": is_weak_representation(result.lrs),
        "p": result.p,
        "epsilon": result.epsilon,
        "components": {
            "coh_plus": result.breakdown.coh_plus,
            "coh_minus": result.breakdown.coh_minus,
            "opposition": result.breakdown.opposition,
            "orthogonality": result.breakdown.orthogonality,
            "confound_independence": result.breakdown.confound_independence,
            "predictivity_gain_norm": result.breakdown.predictivity_gain_norm,
        },
    }


__all__ = [
    "LRSBreakdown",
    "LRSResult",
    "power_mean",
    "compute_lrs",
    "compute_lrs_from_breakdown",
    "compute_lrs_batch",
    "is_strong_representation",
    "is_weak_representation",
    "get_lrs_summary",
]
