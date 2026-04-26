"""
Activation confidence computation.

Implements Def 5.5 (§5.3):
n_A^(k) = Σ_{i∈T} Σ_{π∈Π} sig_A(k, i, π)
C_A^(k) = n_A^(k) / (|T| · |Π|)
"""

from typing import Dict, List, Set, Any
from dataclasses import dataclass

from dcaf.core.defaults import TAU_ACT, TAU_A_DEFAULT
from .significance import percentile_threshold_activation, sig_A


@dataclass
class ActivationConfidenceResult:
    """
    Activation confidence result for a component.

    Attributes:
        C_A: Confidence value ∈ [0, 1]
        n_A: Count of significant (signal, probe) pairs
        total_pairs: Total number of (signal, probe) pairs
        significant_pairs: List of (signal, probe) pairs that were significant
    """
    C_A: float
    n_A: int
    total_pairs: int
    significant_pairs: List[tuple]


def compute_activation_confidence(
    component: str,
    magnitudes_by_signal_probe: Dict[tuple, Dict[str, float]],
    tau_act: float = TAU_ACT,
) -> ActivationConfidenceResult:
    """
    Compute activation confidence for a single component.

    C_A^(k) = n_A^(k) / (|T| · |Π|)

    Args:
        component: Component ID
        magnitudes_by_signal_probe: {(signal, probe): {component: magnitude}}
        tau_act: Significance percentile threshold

    Returns:
        ActivationConfidenceResult
    """
    n_A = 0
    significant_pairs = []

    for (signal, probe), all_magnitudes in magnitudes_by_signal_probe.items():
        if component not in all_magnitudes:
            continue

        # Compute threshold for this (signal, probe) pair
        threshold = percentile_threshold_activation(all_magnitudes, tau_act)

        # Check significance
        if sig_A(component, all_magnitudes[component], threshold):
            n_A += 1
            significant_pairs.append((signal, probe))

    total_pairs = len(magnitudes_by_signal_probe)
    C_A = n_A / total_pairs if total_pairs > 0 else 0.0

    return ActivationConfidenceResult(
        C_A=C_A,
        n_A=n_A,
        total_pairs=total_pairs,
        significant_pairs=significant_pairs,
    )


def compute_all_activation_confidences(
    components: Set[str],
    magnitudes_by_signal_probe: Dict[tuple, Dict[str, float]],
    tau_act: float = TAU_ACT,
) -> Dict[str, ActivationConfidenceResult]:
    """
    Compute activation confidence for all components.

    Args:
        components: Set of component IDs
        magnitudes_by_signal_probe: {(signal, probe): {component: magnitude}}
        tau_act: Significance percentile threshold

    Returns:
        {component: ActivationConfidenceResult}
    """
    return {
        component: compute_activation_confidence(
            component, magnitudes_by_signal_probe, tau_act
        )
        for component in components
    }


def filter_by_activation_confidence(
    results: Dict[str, ActivationConfidenceResult],
    tau_A: float = TAU_A_DEFAULT,
) -> Set[str]:
    """
    Filter components by activation confidence threshold.

    {k : C_A^(k) >= τ_A}

    Args:
        results: {component: ActivationConfidenceResult}
        tau_A: Confidence threshold

    Returns:
        Set of components passing threshold
    """
    return {
        component for component, result in results.items()
        if result.C_A >= tau_A
    }


def get_confidence_summary(
    results: Dict[str, ActivationConfidenceResult],
) -> Dict[str, Any]:
    """
    Summary statistics for activation confidence results.

    Args:
        results: {component: ActivationConfidenceResult}

    Returns:
        Summary statistics dict
    """
    if not results:
        return {"count": 0}

    confidences = [r.C_A for r in results.values()]
    return {
        "count": len(results),
        "mean_C_A": sum(confidences) / len(confidences),
        "max_C_A": max(confidences),
        "min_C_A": min(confidences),
        "significant_count": sum(1 for r in results.values() if r.n_A > 0),
    }


def rank_by_activation_confidence(
    results: Dict[str, ActivationConfidenceResult],
) -> List[tuple]:
    """
    Rank components by activation confidence.

    Args:
        results: {component: ActivationConfidenceResult}

    Returns:
        [(component, result), ...] sorted by C_A descending
    """
    return sorted(
        results.items(),
        key=lambda x: x[1].C_A,
        reverse=True,
    )


__all__ = [
    "ActivationConfidenceResult",
    "compute_activation_confidence",
    "compute_all_activation_confidences",
    "filter_by_activation_confidence",
    "get_confidence_summary",
    "rank_by_activation_confidence",
]
