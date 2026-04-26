"""Unified confidence and candidate selection (§8, Def 8.1-8.5)."""

from dcaf.confidence.triangulation import (
    TriangulationConfig,
    triangulate,
    triangulate_batch,
    compute_domain_contribution,
    compute_domain_deviations,
    compute_domain_disagreement,
    compute_full_diagnostics,
    rank_by_triangulated,
    filter_by_triangulated,
    UnifiedConfidence,
    compute_unified_confidence,
    compute_unified_batch,
    filter_by_unified,
    # Re-exports from domains.base
    DomainType,
    DomainConfidence,
    TriangulatedConfidence,
)
from dcaf.confidence.signals import (
    compute_relevance_confidence,
    SignalDetails,
    aggregate_cross_validated_signals,
)
from dcaf.confidence.thresholds import (
    ThresholdConfig,
    STRICT_THRESHOLDS,
    MODERATE_THRESHOLDS,
    PERMISSIVE_THRESHOLDS,
    EXPLORATORY_THRESHOLDS,
    compute_percentile_threshold,
    compute_adaptive_thresholds,
    compute_elbow_threshold,
    validate_thresholds,
    suggest_thresholds,
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
