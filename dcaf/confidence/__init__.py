"""Unified confidence and candidate selection (sec:unified-confidence)."""

from dcaf.confidence.signals import (
    SignalDetails,
    aggregate_cross_validated_signals,
    compute_relevance_confidence,
)
from dcaf.confidence.thresholds import (
    EXPLORATORY_THRESHOLDS,
    MODERATE_THRESHOLDS,
    PERMISSIVE_THRESHOLDS,
    STRICT_THRESHOLDS,
    ThresholdConfig,
    compute_adaptive_thresholds,
    compute_elbow_threshold,
    compute_percentile_threshold,
    suggest_thresholds,
    validate_thresholds,
)
from dcaf.confidence.triangulation import (
    DomainConfidence,
    # Re-exports from domains.base
    DomainType,
    TriangulatedConfidence,
    TriangulationConfig,
    UnifiedConfidence,
    compute_domain_contribution,
    compute_domain_deviations,
    compute_domain_disagreement,
    compute_full_diagnostics,
    compute_unified_batch,
    compute_unified_confidence,
    filter_by_triangulated,
    filter_by_unified,
    rank_by_triangulated,
    triangulate,
    triangulate_batch,
)

__all__ = [
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
    # Unified confidence
    "UnifiedConfidence",
    "compute_unified_confidence",
    "compute_unified_batch",
    "filter_by_unified",
    # Domain base types (re-exported)
    "DomainType",
    "DomainConfidence",
    "TriangulatedConfidence",
    # Signal/candidate confidence
    "compute_relevance_confidence",
    "SignalDetails",
    "aggregate_cross_validated_signals",
    # Thresholds
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
