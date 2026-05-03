"""
Geometry analysis domain (sec:geometry-analysis): representational structure and linear representation.

Exports the primary public API for geometry analysis. Full per-submodule
exports are available by importing the submodules directly.
"""

from dcaf.domains.geometry.alignment import (
    ClusterMetrics,
    compute_alignment_matrix,
    compute_alignment_matrix_indexed,
    compute_baseline_orthogonality,
    compute_cluster_coherence,
    compute_cluster_metrics,
    compute_cluster_metrics_from_directions,
    compute_cluster_opposition,
    get_alignment_summary,
)
from dcaf.domains.geometry.confidence import (
    GeometryConfidenceResult,
    compute_all_geometry_confidences,
    compute_geometry_confidence,
    compute_geometry_confidence_simple,
    filter_by_geometry_confidence,
    get_component_breakdown,
    get_geometry_confidence_summary,
    rank_by_geometry_confidence,
)
from dcaf.domains.geometry.confound import (
    ConfoundAnalysis,
    compute_confound_analysis,
    compute_confound_independence,
    compute_direction_overlap,
    extract_confound_direction,
    is_clean,
    is_contaminated,
)
from dcaf.domains.geometry.directions import (
    DirectionDynamics,
    DirectionMethod,
    WhitenedSVDResult,
    aggregate_directions,
    compute_direction_dynamics,
    compute_direction_emergence,
    compute_direction_rotation,
    compute_pooled_covariance,
    extract_contrastive_direction,
    extract_contrastive_directions_batch,
    extract_dim_direction,
    extract_whitened_svd_directions,
)
from dcaf.domains.geometry.generalization import (
    GeneralizationResult,
    compute_generalization,
    compute_generalization_ratio,
    compute_generalization_simple,
    get_generalization_summary,
    is_generalizable,
    is_overfitting,
)
from dcaf.domains.geometry.lrs import (
    LRSBreakdown,
    LRSResult,
    compute_lrs,
    compute_lrs_batch,
    compute_lrs_from_breakdown,
    get_lrs_summary,
    is_strong_representation,
    is_weak_representation,
    power_mean,
)
from dcaf.domains.geometry.nonlinear import (
    NonlinearDiagnostics,
    compute_nonlinear_diagnostics,
    compute_pacmap_on_activations,
    compute_pacmap_on_deltas,
    compute_procrustes_alignment,
)
from dcaf.domains.geometry.predictivity import (
    PredictivityResult,
    compute_auc,
    compute_predictivity,
    compute_predictivity_at_threshold,
    compute_predictivity_gain,
    compute_predictivity_gain_batch,
    normalize_predictivity_gain,
)
from dcaf.domains.geometry.probing import (
    ProbingResults,
    compute_kernel_lda,
    compute_polynomial_probing,
)

__all__ = [
    # directions (def:contrastive-direction; def:direction-emergence; def:direction-rotation)
    "DirectionMethod",
    "DirectionDynamics",
    "WhitenedSVDResult",
    "compute_pooled_covariance",
    "extract_dim_direction",
    "extract_whitened_svd_directions",
    "extract_contrastive_direction",
    "extract_contrastive_directions_batch",
    "compute_direction_emergence",
    "compute_direction_rotation",
    "compute_direction_dynamics",
    "aggregate_directions",
    # alignment (def:alignment-matrix; def:opposition-verification; def:lrs-components-from-alignment)
    "ClusterMetrics",
    "compute_alignment_matrix",
    "compute_alignment_matrix_indexed",
    "compute_cluster_coherence",
    "compute_cluster_opposition",
    "compute_baseline_orthogonality",
    "compute_cluster_metrics",
    "compute_cluster_metrics_from_directions",
    "get_alignment_summary",
    # confound (def:confound-direction; def:confound-independence)
    "ConfoundAnalysis",
    "extract_confound_direction",
    "compute_direction_overlap",
    "compute_confound_independence",
    "compute_confound_analysis",
    "is_contaminated",
    "is_clean",
    # predictivity (def:direction-predictivity; def:predictivity-gain)
    "PredictivityResult",
    "compute_auc",
    "compute_predictivity",
    "compute_predictivity_gain",
    "compute_predictivity_gain_batch",
    "normalize_predictivity_gain",
    "compute_predictivity_at_threshold",
    # generalization (def:generalization)
    "GeneralizationResult",
    "compute_generalization_ratio",
    "compute_generalization",
    "compute_generalization_simple",
    "is_generalizable",
    "is_overfitting",
    "get_generalization_summary",
    # lrs (def:lrs)
    "LRSBreakdown",
    "LRSResult",
    "power_mean",
    "compute_lrs",
    "compute_lrs_from_breakdown",
    "compute_lrs_batch",
    "is_strong_representation",
    "is_weak_representation",
    "get_lrs_summary",
    # confidence (def:geometric-confidence)
    "GeometryConfidenceResult",
    "compute_geometry_confidence",
    "compute_geometry_confidence_simple",
    "compute_all_geometry_confidences",
    "filter_by_geometry_confidence",
    "get_geometry_confidence_summary",
    "rank_by_geometry_confidence",
    "get_component_breakdown",
    # nonlinear (def:nonlinear-diagnostic-trigger; def:pacmap-diagnostics; def:procrustes)
    "NonlinearDiagnostics",
    "compute_pacmap_on_activations",
    "compute_pacmap_on_deltas",
    "compute_procrustes_alignment",
    "compute_nonlinear_diagnostics",
    # probing (def:polynomial-probing; def:kernel-lda)
    "ProbingResults",
    "compute_kernel_lda",
    "compute_polynomial_probing",
]
