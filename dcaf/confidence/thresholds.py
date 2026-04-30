"""
Confidence thresholds and domain analysis for candidate filtering (§8).

Implements domain disagreement analysis (Def 13.1-13.3) and threshold
configuration for candidate set construction (Def 8.4-8.5).

Thresholds:
- τ_W: Weight confidence threshold (default 0.3; spec tuning range 0.3-0.5)
- τ_A: Activation confidence threshold (default 0.3; spec tuning range 0.3-0.5)
- τ_G: Geometry confidence threshold (default 0.3; spec tuning range 0.3-0.5)
- τ_unified: Candidate threshold (default 0.3)

Default thresholds match the formal specification. Tune upward for stricter
candidate sets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Sequence

from dcaf.core.defaults import (
    TAU_W_DEFAULT, TAU_A_DEFAULT, TAU_G_DEFAULT, TAU_UNIFIED_DEFAULT,
    TAU_MIN, TAU_MAX, DEFAULT_MISSING_CONFIDENCE,
    PERCENTILE_FILTER, TOP_K_CANDIDATES,
)
from dcaf.domains.base import DomainType


@dataclass
class ThresholdConfig:
    """
    Threshold configuration for candidate filtering.

    Attributes:
        tau_W: Weight domain threshold
        tau_A: Activation domain threshold
        tau_G: Geometry domain threshold
        tau_unified: Triangulated confidence threshold
    """
    tau_W: float = TAU_W_DEFAULT
    tau_A: float = TAU_A_DEFAULT
    tau_G: float = TAU_G_DEFAULT
    tau_unified: float = TAU_UNIFIED_DEFAULT

    def __post_init__(self):
        for name, val in [
            ("tau_W", self.tau_W),
            ("tau_A", self.tau_A),
            ("tau_G", self.tau_G),
            ("tau_unified", self.tau_unified),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def get_threshold(self, domain: DomainType) -> float:
        """Get threshold for a specific domain."""
        if domain == DomainType.WEIGHT:
            return self.tau_W
        elif domain == DomainType.ACTIVATION:
            return self.tau_A
        elif domain == DomainType.GEOMETRY:
            return self.tau_G
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "tau_W": self.tau_W,
            "tau_A": self.tau_A,
            "tau_G": self.tau_G,
            "tau_unified": self.tau_unified,
        }


# Preset configurations
STRICT_THRESHOLDS = ThresholdConfig(
    tau_W=TAU_MAX,
    tau_A=TAU_MAX,
    tau_G=TAU_MAX,
    tau_unified=TAU_MAX,
)

MODERATE_THRESHOLDS = ThresholdConfig(
    tau_W=0.4,
    tau_A=0.4,
    tau_G=0.4,
    tau_unified=0.4,
)

PERMISSIVE_THRESHOLDS = ThresholdConfig(
    tau_W=TAU_MIN,
    tau_A=TAU_MIN,
    tau_G=TAU_MIN,
    tau_unified=TAU_MIN,
)

EXPLORATORY_THRESHOLDS = ThresholdConfig(
    tau_W=TAU_W_DEFAULT,
    tau_A=TAU_A_DEFAULT,
    tau_G=TAU_G_DEFAULT,
    tau_unified=TAU_UNIFIED_DEFAULT,
)


def compute_percentile_threshold(
    values: Sequence[float],
    percentile: float = 50.0,
) -> float:
    """
    Compute threshold at a given percentile of values.

    Args:
        values: Sequence of confidence values
        percentile: Percentile (0-100) to use as threshold

    Returns:
        Value at the given percentile
    """
    if not values:
        return DEFAULT_MISSING_CONFIDENCE

    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_adaptive_thresholds(
    C_W_values: Sequence[float],
    C_A_values: Sequence[float],
    C_G_values: Sequence[float],
    target_percentile: float = PERCENTILE_FILTER,
) -> ThresholdConfig:
    """
    Compute adaptive thresholds based on score distributions.

    Sets threshold at the target_percentile of each domain's scores,
    ensuring only top (100 - target_percentile)% of candidates pass.

    Args:
        C_W_values: Weight confidence scores
        C_A_values: Activation confidence scores
        C_G_values: Geometry confidence scores
        target_percentile: Percentile for threshold (default 85)

    Returns:
        ThresholdConfig with adaptive thresholds
    """
    tau_W = compute_percentile_threshold(C_W_values, target_percentile)
    tau_A = compute_percentile_threshold(C_A_values, target_percentile)
    tau_G = compute_percentile_threshold(C_G_values, target_percentile)

    # Unified threshold is the geometric mean
    tau_unified = (tau_W * tau_A * tau_G) ** (1/3)

    return ThresholdConfig(
        tau_W=tau_W,
        tau_A=tau_A,
        tau_G=tau_G,
        tau_unified=tau_unified,
    )


def compute_elbow_threshold(
    values: Sequence[float],
    min_threshold: float = TAU_MIN,
    max_threshold: float = TAU_MAX,
) -> float:
    """
    Find threshold using elbow/knee detection on sorted values.

    Useful when the distribution has a natural break point.

    Args:
        values: Sequence of confidence values
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold

    Returns:
        Threshold at the elbow point
    """
    if len(values) < 3:
        return DEFAULT_MISSING_CONFIDENCE

    sorted_vals = sorted(values, reverse=True)
    n = len(sorted_vals)

    # Simple elbow detection: find max second derivative
    max_curvature = 0
    elbow_idx = n // 2

    for i in range(1, n - 1):
        # Approximate second derivative
        curvature = abs(sorted_vals[i-1] - 2*sorted_vals[i] + sorted_vals[i+1])
        if curvature > max_curvature:
            max_curvature = curvature
            elbow_idx = i

    threshold = sorted_vals[elbow_idx]
    return max(min_threshold, min(max_threshold, threshold))


def validate_thresholds(config: ThresholdConfig) -> List[str]:
    """
    Validate threshold configuration and return warnings.

    Args:
        config: ThresholdConfig to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Check for very low thresholds
    for name, val in [
        ("tau_W", config.tau_W),
        ("tau_A", config.tau_A),
        ("tau_G", config.tau_G),
    ]:
        if val < 0.1:
            warnings.append(f"{name}={val:.2f} is very low, may include many false positives")

    # Check for very high thresholds
    for name, val in [
        ("tau_W", config.tau_W),
        ("tau_A", config.tau_A),
        ("tau_G", config.tau_G),
    ]:
        if val > 0.8:
            warnings.append(f"{name}={val:.2f} is very high, may miss true positives")

    # Check for large imbalance
    vals = [config.tau_W, config.tau_A, config.tau_G]
    if max(vals) - min(vals) > 0.3:
        warnings.append(f"Large threshold imbalance: W={config.tau_W:.2f}, A={config.tau_A:.2f}, G={config.tau_G:.2f}")

    return warnings


def suggest_thresholds(
    C_W_values: Sequence[float],
    C_A_values: Optional[Sequence[float]] = None,
    C_G_values: Optional[Sequence[float]] = None,
    target_candidates: int = TOP_K_CANDIDATES,
) -> ThresholdConfig:
    """
    Suggest thresholds to achieve approximately target_candidates.

    Args:
        C_W_values: Weight confidence scores (required)
        C_A_values: Activation confidence scores (optional)
        C_G_values: Geometry confidence scores (optional)
        target_candidates: Target number of candidates to retain

    Returns:
        ThresholdConfig that would yield ~target_candidates
    """
    n = len(C_W_values)
    if n == 0:
        return MODERATE_THRESHOLDS

    # Compute percentile that gives target count
    target_pct = 100 * (1 - target_candidates / n)
    target_pct = max(50, min(95, target_pct))  # Clamp to reasonable range

    # Weight domain is required
    tau_W = compute_percentile_threshold(C_W_values, target_pct)

    # Use adaptive or default for other domains
    tau_A = compute_percentile_threshold(C_A_values, target_pct) if C_A_values else DEFAULT_MISSING_CONFIDENCE
    tau_G = compute_percentile_threshold(C_G_values, target_pct) if C_G_values else DEFAULT_MISSING_CONFIDENCE

    return ThresholdConfig(
        tau_W=max(TAU_MIN, tau_W),
        tau_A=max(TAU_MIN, tau_A),
        tau_G=max(TAU_MIN, tau_G),
        tau_unified=TAU_UNIFIED_DEFAULT,
    )


__all__ = [
    "ThresholdConfig",
    "STRICT_THRESHOLDS",
    "MODERATE_THRESHOLDS",
    "PERMISSIVE_THRESHOLDS",
    "EXPLORATORY_THRESHOLDS",
    "compute_percentile_threshold",
    "compute_adaptive_thresholds",
    "compute_elbow_threshold",
    "validate_thresholds",
    "suggest_thresholds",
]
