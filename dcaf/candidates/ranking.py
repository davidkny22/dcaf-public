"""
Multi-domain ranking for candidate prioritization.

Combines confidence scores from all three domains to rank candidates
for further analysis (ablation testing, circuit construction).
"""

from typing import Dict, Set, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from dcaf.core.defaults import DEFAULT_MISSING_CONFIDENCE
from dcaf.confidence.triangulation import triangulate, TriangulatedConfidence


class RankingMethod(str, Enum):
    """Methods for combining domain scores."""
    TRIANGULATED = "triangulated"  # Use triangulation formula
    GEOMETRIC_MEAN = "geometric_mean"  # Simple geometric mean
    ARITHMETIC_MEAN = "arithmetic_mean"  # Simple arithmetic mean
    MINIMUM = "minimum"  # Conservative: min of all domains
    WEIGHT_PRIORITY = "weight_priority"  # Weight-first tiebreaker


@dataclass
class RankedCandidate:
    """A candidate with ranking information."""
    id: Any
    rank: int
    score: float
    C_W: Optional[float] = None
    C_A: Optional[float] = None
    C_G: Optional[float] = None
    method: RankingMethod = RankingMethod.TRIANGULATED


def compute_combined_score(
    C_W: Optional[float],
    C_A: Optional[float],
    C_G: Optional[float],
    method: RankingMethod = RankingMethod.TRIANGULATED,
) -> float:
    """
    Compute combined score from domain confidences.

    Args:
        C_W: Weight confidence (None = neutral)
        C_A: Activation confidence (None = neutral)
        C_G: Geometry confidence (None = neutral)
        method: Combination method

    Returns:
        Combined score ∈ [0, 1]
    """
    # Default missing to neutral
    w = C_W if C_W is not None else DEFAULT_MISSING_CONFIDENCE
    a = C_A if C_A is not None else DEFAULT_MISSING_CONFIDENCE
    g = C_G if C_G is not None else DEFAULT_MISSING_CONFIDENCE

    if method == RankingMethod.TRIANGULATED:
        tri = triangulate(C_W=w, C_A=a, C_G=g)
        return tri.value if tri else 0.0

    elif method == RankingMethod.GEOMETRIC_MEAN:
        return (w * a * g) ** (1/3)

    elif method == RankingMethod.ARITHMETIC_MEAN:
        return (w + a + g) / 3

    elif method == RankingMethod.MINIMUM:
        return min(w, a, g)

    elif method == RankingMethod.WEIGHT_PRIORITY:
        # Weight is primary, others are tiebreakers
        return w + 0.01 * a + 0.0001 * g

    else:
        return (w * a * g) ** (1/3)


def rank_candidates(
    weight_confidences: Dict[Any, float],
    activation_confidences: Optional[Dict[Any, float]] = None,
    geometry_confidences: Optional[Dict[Any, float]] = None,
    component_map: Optional[Dict[Any, Any]] = None,
    method: RankingMethod = RankingMethod.TRIANGULATED,
    top_k: Optional[int] = None,
) -> List[RankedCandidate]:
    """
    Rank candidates by combined confidence score.

    Args:
        weight_confidences: {param_id: C_W} mapping
        activation_confidences: {component_id: C_A} mapping (optional)
        geometry_confidences: {component_id: C_G} mapping (optional)
        component_map: {param_id: component_id} mapping (optional)
        method: Ranking method
        top_k: Return only top k (None = all)

    Returns:
        List of RankedCandidate sorted by score descending
    """
    candidates = []

    for param_id, c_w in weight_confidences.items():
        # Look up component-level confidences
        c_a = None
        c_g = None

        if component_map and (activation_confidences or geometry_confidences):
            component = component_map.get(param_id)
            if component:
                if activation_confidences:
                    c_a = activation_confidences.get(component)
                if geometry_confidences:
                    c_g = geometry_confidences.get(component)

        score = compute_combined_score(c_w, c_a, c_g, method)

        candidates.append(RankedCandidate(
            id=param_id,
            rank=0,  # Will be set after sorting
            score=score,
            C_W=c_w,
            C_A=c_a,
            C_G=c_g,
            method=method,
        ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Assign ranks
    for i, c in enumerate(candidates):
        c.rank = i + 1

    if top_k is not None:
        return candidates[:top_k]

    return candidates


def rank_components(
    activation_confidences: Dict[Any, float],
    geometry_confidences: Dict[Any, float],
    method: RankingMethod = RankingMethod.GEOMETRIC_MEAN,
    top_k: Optional[int] = None,
) -> List[Tuple[Any, float]]:
    """
    Rank components by activation and geometry confidence.

    Args:
        activation_confidences: {component_id: C_A} mapping
        geometry_confidences: {component_id: C_G} mapping
        method: Ranking method
        top_k: Return only top k

    Returns:
        [(component_id, score), ...] sorted descending
    """
    all_components = set(activation_confidences.keys()) | set(geometry_confidences.keys())

    scores = []
    for component in all_components:
        c_a = activation_confidences.get(component)
        c_g = geometry_confidences.get(component)

        # For component ranking, weight is not used
        score = compute_combined_score(None, c_a, c_g, method)
        scores.append((component, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return scores[:top_k]

    return scores


def get_ranking_summary(
    ranked: List[RankedCandidate],
) -> Dict[str, Any]:
    """
    Get summary statistics for a ranking.

    Args:
        ranked: List of RankedCandidate

    Returns:
        Summary statistics
    """
    if not ranked:
        return {"count": 0}

    scores = [c.score for c in ranked]
    c_w_vals = [c.C_W for c in ranked if c.C_W is not None]
    c_a_vals = [c.C_A for c in ranked if c.C_A is not None]
    c_g_vals = [c.C_G for c in ranked if c.C_G is not None]

    return {
        "count": len(ranked),
        "method": ranked[0].method.value,
        "score_mean": sum(scores) / len(scores),
        "score_max": max(scores),
        "score_min": min(scores),
        "C_W_mean": sum(c_w_vals) / len(c_w_vals) if c_w_vals else None,
        "C_A_mean": sum(c_a_vals) / len(c_a_vals) if c_a_vals else None,
        "C_G_mean": sum(c_g_vals) / len(c_g_vals) if c_g_vals else None,
    }


__all__ = [
    "RankingMethod",
    "RankedCandidate",
    "compute_combined_score",
    "rank_candidates",
    "rank_components",
    "get_ranking_summary",
]
