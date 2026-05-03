"""
Geometry confidence computation.

Implements def:geometric-confidence:
C_G^(k) = LRS(k) · gen(k)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from dcaf.core.defaults import TAU_G_DEFAULT

from .generalization import GeneralizationResult
from .lrs import LRSResult


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class GeometryConfidenceResult:
    """
    Geometry confidence result for a component.

    Attributes:
        C_G: Geometry confidence value ∈ [0, 1]
        lrs: Linear Representation Score
        gen: Generalization score
        lrs_result: Full LRS result with breakdown
        gen_result: Full generalization result
    """
    C_G: float
    lrs: float
    gen: float
    lrs_result: LRSResult
    gen_result: GeneralizationResult


def compute_geometry_confidence(
    lrs_result: LRSResult,
    gen_result: GeneralizationResult,
) -> GeometryConfidenceResult:
    """
    Compute geometry confidence for a component.

    C_G^(k) = LRS(k) · gen(k)

    Args:
        lrs_result: LRS computation result
        gen_result: Generalization computation result

    Returns:
        GeometryConfidenceResult
    """
    lrs = _clamp_unit(lrs_result.lrs)
    gen = _clamp_unit(gen_result.gen)
    C_G = _clamp_unit(lrs * gen)

    return GeometryConfidenceResult(
        C_G=C_G,
        lrs=lrs,
        gen=gen,
        lrs_result=lrs_result,
        gen_result=gen_result,
    )


def compute_geometry_confidence_simple(
    lrs: float,
    gen: float,
) -> float:
    """
    Compute geometry confidence from scalar values.

    C_G^(k) = LRS(k) · gen(k)

    Args:
        lrs: LRS score
        gen: Generalization score

    Returns:
        Geometry confidence value
    """
    return _clamp_unit(_clamp_unit(lrs) * _clamp_unit(gen))


def compute_all_geometry_confidences(
    lrs_results: Dict[str, LRSResult],
    gen_results: Dict[str, GeneralizationResult],
) -> Dict[str, GeometryConfidenceResult]:
    """
    Compute geometry confidence for all components.

    Args:
        lrs_results: {component: LRSResult}
        gen_results: {component: GeneralizationResult}

    Returns:
        {component: GeometryConfidenceResult}
    """
    results = {}

    for component in lrs_results.keys():
        if component not in gen_results:
            continue

        results[component] = compute_geometry_confidence(
            lrs_results[component],
            gen_results[component],
        )

    return results


def filter_by_geometry_confidence(
    results: Dict[str, GeometryConfidenceResult],
    tau_G: float = TAU_G_DEFAULT,
) -> Set[str]:
    """
    Filter components by geometry confidence threshold.

    {k : C_G^(k) >= τ_G}

    Args:
        results: {component: GeometryConfidenceResult}
        tau_G: Geometry confidence threshold

    Returns:
        Set of components passing threshold
    """
    return {
        component for component, result in results.items()
        if result.C_G >= tau_G
    }


def get_geometry_confidence_summary(
    results: Dict[str, GeometryConfidenceResult],
) -> Dict[str, Any]:
    """
    Summary statistics for geometry confidence results.

    Args:
        results: {component: GeometryConfidenceResult}

    Returns:
        Summary statistics dict
    """
    if not results:
        return {"count": 0}

    confidences = [r.C_G for r in results.values()]
    lrs_scores = [r.lrs for r in results.values()]
    gen_scores = [r.gen for r in results.values()]

    return {
        "count": len(results),
        "mean_C_G": sum(confidences) / len(confidences),
        "max_C_G": max(confidences),
        "min_C_G": min(confidences),
        "mean_lrs": sum(lrs_scores) / len(lrs_scores),
        "mean_gen": sum(gen_scores) / len(gen_scores),
        "above_threshold_count": sum(1 for c in confidences if c >= TAU_G_DEFAULT),
    }


def rank_by_geometry_confidence(
    results: Dict[str, GeometryConfidenceResult],
) -> List[tuple]:
    """
    Rank components by geometry confidence.

    Args:
        results: {component: GeometryConfidenceResult}

    Returns:
        [(component, result), ...] sorted by C_G descending
    """
    return sorted(
        results.items(),
        key=lambda x: x[1].C_G,
        reverse=True,
    )


def get_component_breakdown(
    result: GeometryConfidenceResult,
) -> Dict[str, Any]:
    """
    Get detailed breakdown for a component's geometry confidence.

    Args:
        result: GeometryConfidenceResult

    Returns:
        Detailed breakdown dict
    """
    return {
        "C_G": result.C_G,
        "lrs": result.lrs,
        "gen": result.gen,
        "lrs_breakdown": {
            "coh_plus": result.lrs_result.breakdown.coh_plus,
            "coh_minus": result.lrs_result.breakdown.coh_minus,
            "opposition": result.lrs_result.breakdown.opposition,
            "orthogonality": result.lrs_result.breakdown.orthogonality,
            "confound_independence": result.lrs_result.breakdown.confound_independence,
            "predictivity_gain_norm": result.lrs_result.breakdown.predictivity_gain_norm,
        },
        "gen_breakdown": {
            "gap": result.gen_result.gap,
            "mean_pred_within": result.gen_result.mean_pred_within,
            "mean_pred_ood": result.gen_result.mean_pred_ood,
        },
    }


__all__ = [
    "GeometryConfidenceResult",
    "compute_geometry_confidence",
    "compute_geometry_confidence_simple",
    "compute_all_geometry_confidences",
    "filter_by_geometry_confidence",
    "get_geometry_confidence_summary",
    "rank_by_geometry_confidence",
    "get_component_breakdown",
]
