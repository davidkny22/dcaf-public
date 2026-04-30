"""
Triangulated confidence from multiple measurement domains (§8).

Implements §8 (Unified Confidence and Candidate Selection):
  Base triangulated confidence (§8 Def "Base Triangulated Confidence"):
    C_base = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^(1/(w+2))

  Multi-path discovery bonus (§8 Def "Multi-Path Discovery Bonus"):
    bonus = β_path * max(0, path_count - 1)

  Final unified confidence (§8 Def "Final Unified Confidence"):
    C = min(1, C_base + bonus)

Also implements domain disagreement and deviation analysis (§13):
  Disagree(k) = Var(C_W, C_A, C_G) = (1/3) * sum((C_d - C_mean)^2)
  dev_d = C_d - C_mean
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, List, Any

from dcaf.core.defaults import (
    W_DISCOVERY, EPSILON_TRI, DEFAULT_MISSING_CONFIDENCE, TAU_UNIFIED_DEFAULT,
    BETA_PATH,
)
# Re-export from base for convenience
from dcaf.domains.base import (
    DomainType,
    DomainConfidence,
    TriangulatedConfidence,
)


@dataclass
class TriangulationConfig:
    """
    Configuration for confidence triangulation.

    Attributes:
        weight_power: Power for weight domain
        epsilon: Smoothing constant
        require_all_domains: If True, return None when any domain missing
        default_missing: Default value for missing domains
    """
    weight_power: int = W_DISCOVERY
    epsilon: float = EPSILON_TRI
    require_all_domains: bool = False
    default_missing: float = DEFAULT_MISSING_CONFIDENCE


def triangulate(
    C_W: Optional[float] = None,
    C_A: Optional[float] = None,
    C_G: Optional[float] = None,
    config: Optional[TriangulationConfig] = None,
) -> Optional[TriangulatedConfidence]:
    """
    Compute triangulated confidence from domain scores.

    Formula: C = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^(1/(w+2))

    Args:
        C_W: Weight domain confidence (or None)
        C_A: Activation domain confidence (or None)
        C_G: Geometry domain confidence (or None)
        config: Triangulation configuration

    Returns:
        TriangulatedConfidence, or None if require_all_domains and one is missing
    """
    if config is None:
        config = TriangulationConfig()

    # Check if all domains required
    if config.require_all_domains:
        if C_W is None or C_A is None or C_G is None:
            return None

    return TriangulatedConfidence.compute(
        C_W=C_W,
        C_A=C_A,
        C_G=C_G,
        weight_power=config.weight_power,
        epsilon=config.epsilon,
    )


def triangulate_batch(
    candidates: Dict[Any, Dict[str, Optional[float]]],
    config: Optional[TriangulationConfig] = None,
) -> Dict[Any, TriangulatedConfidence]:
    """
    Compute triangulated confidence for multiple candidates.

    Args:
        candidates: {candidate_id: {"C_W": val, "C_A": val, "C_G": val}}
        config: Triangulation configuration

    Returns:
        {candidate_id: TriangulatedConfidence}
    """
    if config is None:
        config = TriangulationConfig()

    results = {}
    for cid, scores in candidates.items():
        tri = triangulate(
            C_W=scores.get("C_W"),
            C_A=scores.get("C_A"),
            C_G=scores.get("C_G"),
            config=config,
        )
        if tri is not None:
            results[cid] = tri

    return results


def compute_domain_contribution(
    C_W: float,
    C_A: float,
    C_G: float,
    config: Optional[TriangulationConfig] = None,
) -> Dict[str, float]:
    """
    Compute relative contribution of each domain to final score.

    Useful for understanding which domain is driving the confidence.

    Args:
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence
        config: Triangulation configuration

    Returns:
        {"weight": pct, "activation": pct, "geometry": pct}
    """
    if config is None:
        config = TriangulationConfig()

    eps = config.epsilon
    w = config.weight_power

    # Decompose the multiplicative formula in log space:
    # log(C_base) is proportional to w*log(C_W+eps)+log(C_A+eps)+log(C_G+eps).
    log_w = w * math.log(max(C_W + eps, eps))
    log_a = math.log(max(C_A + eps, eps))
    log_g = math.log(max(C_G + eps, eps))

    total = abs(log_w) + abs(log_a) + abs(log_g)

    if total <= 0:
        return {"weight": 0.33, "activation": 0.33, "geometry": 0.34}

    return {
        "weight": abs(log_w) / total,
        "activation": abs(log_a) / total,
        "geometry": abs(log_g) / total,
    }


def compute_domain_deviations(
    C_W: float,
    C_A: float,
    C_G: float,
) -> Dict[str, float]:
    """
    Compute deviation of each domain from the mean.

    The deviations sum to 0 by construction.

    Args:
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence

    Returns:
        {"weight": dev, "activation": dev, "geometry": dev}
    """
    mean = (C_W + C_A + C_G) / 3.0
    return {
        "weight": C_W - mean,
        "activation": C_A - mean,
        "geometry": C_G - mean,
    }


def compute_domain_disagreement(
    C_W: float,
    C_A: float,
    C_G: float,
) -> float:
    """
    Compute disagreement (variance) across domains.

    Interpretation:
    - < 0.02: Strong convergent evidence
    - 0.02–0.08: Some domain variation, worth investigating
    - > 0.08: Conflicting signals, prioritize for manual analysis

    Args:
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence

    Returns:
        Variance across the three domain scores
    """
    mean = (C_W + C_A + C_G) / 3.0
    variance = (
        (C_W - mean) ** 2 +
        (C_A - mean) ** 2 +
        (C_G - mean) ** 2
    ) / 3.0
    return variance


def compute_full_diagnostics(
    C_W: float,
    C_A: float,
    C_G: float,
    config: Optional[TriangulationConfig] = None,
) -> Dict[str, Any]:
    """
    Compute all domain diagnostics combined.

    Returns:
        Dict with contributions, deviations, disagreement, and mean
    """
    return {
        "contributions": compute_domain_contribution(C_W, C_A, C_G, config),
        "deviations": compute_domain_deviations(C_W, C_A, C_G),
        "disagreement": compute_domain_disagreement(C_W, C_A, C_G),
        "mean": (C_W + C_A + C_G) / 3.0,
    }


def rank_by_triangulated(
    results: Dict[Any, TriangulatedConfidence],
    top_k: Optional[int] = None,
) -> List[tuple]:
    """
    Rank candidates by triangulated confidence.

    Args:
        results: {candidate_id: TriangulatedConfidence}
        top_k: Return only top k (None = all)

    Returns:
        [(candidate_id, TriangulatedConfidence), ...] sorted descending
    """
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].value,
        reverse=True,
    )

    if top_k is not None:
        return sorted_results[:top_k]

    return sorted_results


def filter_by_triangulated(
    results: Dict[Any, TriangulatedConfidence],
    threshold: float = TAU_UNIFIED_DEFAULT,
) -> Dict[Any, TriangulatedConfidence]:
    """
    Filter candidates by triangulated confidence threshold.

    Args:
        results: {candidate_id: TriangulatedConfidence}
        threshold: Minimum confidence to pass

    Returns:
        Filtered dict with only passing candidates
    """
    return {
        cid: tri for cid, tri in results.items()
        if tri.value >= threshold
    }


# =============================================================================
# Unified Confidence (with multi-path bonus)
# =============================================================================

@dataclass
class UnifiedConfidence:
    """
    Unified confidence combining triangulation with multi-path discovery bonus.

    C_unified = min(1, C_base + bonus)

    Where:
    - C_base = triangulated confidence
    - bonus = β_path * max(0, path_count - 1)

    Attributes:
        value: Final unified confidence ∈ [0, 1]
        C_base: Base triangulated confidence
        bonus: Multi-path discovery bonus
        path_count: Number of discovery paths (1, 2, or 3)
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence
    """
    value: float
    C_base: float
    bonus: float
    path_count: int
    C_W: Optional[float] = None
    C_A: Optional[float] = None
    C_G: Optional[float] = None

    @classmethod
    def compute(
        cls,
        C_W: Optional[float] = None,
        C_A: Optional[float] = None,
        C_G: Optional[float] = None,
        path_count: int = 1,
        beta_path: float = BETA_PATH,
        config: Optional[TriangulationConfig] = None,
    ) -> "UnifiedConfidence":
        """
        Compute unified confidence with multi-path bonus.

        Args:
            C_W: Weight domain confidence
            C_A: Activation domain confidence
            C_G: Geometry domain confidence
            path_count: Number of discovery paths (1, 2, or 3)
            beta_path: Multi-path bonus weight
            config: Triangulation configuration

        Returns:
            UnifiedConfidence with computed value
        """
        # Compute base triangulated confidence
        tri = triangulate(C_W, C_A, C_G, config)
        C_base = tri.value if tri is not None else 0.0

        # Compute multi-path bonus
        bonus = beta_path * max(0, path_count - 1)

        # Unified confidence
        value = min(1.0, C_base + bonus)

        return cls(
            value=value,
            C_base=C_base,
            bonus=bonus,
            path_count=path_count,
            C_W=C_W,
            C_A=C_A,
            C_G=C_G,
        )

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return (
            f"UnifiedConfidence(C={self.value:.3f}, "
            f"base={self.C_base:.3f}, bonus={self.bonus:.3f}, "
            f"paths={self.path_count})"
        )


def compute_unified_confidence(
    C_W: Optional[float] = None,
    C_A: Optional[float] = None,
    C_G: Optional[float] = None,
    path_count: int = 1,
    beta_path: float = BETA_PATH,
    config: Optional[TriangulationConfig] = None,
) -> UnifiedConfidence:
    """
    Compute unified confidence with multi-path bonus.

    C_unified = min(1, C_base + bonus)

    Args:
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence
        path_count: Number of discovery paths
        beta_path: Multi-path bonus weight
        config: Triangulation configuration

    Returns:
        UnifiedConfidence
    """
    return UnifiedConfidence.compute(
        C_W=C_W,
        C_A=C_A,
        C_G=C_G,
        path_count=path_count,
        beta_path=beta_path,
        config=config,
    )


def compute_unified_batch(
    candidates: Dict[Any, Dict[str, Any]],
    beta_path: float = BETA_PATH,
    config: Optional[TriangulationConfig] = None,
) -> Dict[Any, UnifiedConfidence]:
    """
    Compute unified confidence for multiple candidates.

    Args:
        candidates: {candidate_id: {"C_W": val, "C_A": val, "C_G": val, "path_count": n}}
        beta_path: Multi-path bonus weight
        config: Triangulation configuration

    Returns:
        {candidate_id: UnifiedConfidence}
    """
    results = {}
    for cid, data in candidates.items():
        results[cid] = compute_unified_confidence(
            C_W=data.get("C_W"),
            C_A=data.get("C_A"),
            C_G=data.get("C_G"),
            path_count=data.get("path_count", 1),
            beta_path=beta_path,
            config=config,
        )
    return results


def filter_by_unified(
    results: Dict[Any, UnifiedConfidence],
    threshold: float = TAU_UNIFIED_DEFAULT,
) -> Dict[Any, UnifiedConfidence]:
    """
    Filter candidates by unified confidence threshold.

    Args:
        results: {candidate_id: UnifiedConfidence}
        threshold: Minimum confidence to pass

    Returns:
        Filtered dict with only passing candidates
    """
    return {
        cid: uc for cid, uc in results.items()
        if uc.value >= threshold
    }


__all__ = [
    # Re-exports from base
    "DomainType",
    "DomainConfidence",
    "TriangulatedConfidence",
    # Triangulation
    "TriangulationConfig",
    "triangulate",
    "triangulate_batch",
    "compute_domain_contribution",
    "compute_domain_deviations",
    "compute_domain_disagreement",
    "compute_full_diagnostics",
    "rank_by_triangulated",
    "filter_by_triangulated",
    # Unified confidence (with multi-path bonus)
    "UnifiedConfidence",
    "compute_unified_confidence",
    "compute_unified_batch",
    "filter_by_unified",
]
