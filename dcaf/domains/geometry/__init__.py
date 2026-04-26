"""
Geometry analysis domain (§6): representational structure and linear representation.

Exports the primary public API for geometry analysis. Full per-submodule
exports are available by importing the submodules directly.
"""

from dcaf.domains.geometry.directions import (
    DirectionDynamics,
    compute_pooled_covariance,
    extract_contrastive_direction,
    extract_contrastive_directions_batch,
    compute_direction_emergence,
    compute_direction_rotation,
    compute_direction_dynamics,
    aggregate_directions,
)
from dcaf.domains.geometry.alignment import (
    ClusterMetrics,
    compute_alignment_matrix,
    compute_alignment_matrix_indexed,
    compute_cluster_coherence,
    compute_cluster_opposition,
    compute_baseline_orthogonality,
    compute_cluster_metrics,
    compute_cluster_metrics_from_directions,
    get_alignment_summary,
)
from dcaf.domains.geometry.confound import (
    ConfoundAnalysis,
    extract_confound_direction,
    compute_direction_overlap,
    compute_confound_independence,
    compute_confound_analysis,
    is_contaminated,
    is_clean,
)
from dcaf.domains.geometry.predictivity import (
    PredictivityResult,
    compute_auc,
    compute_predictivity,
    compute_predictivity_gain,
    compute_predictivity_gain_batch,
    normalize_predictivity_gain,
    compute_predictivity_at_threshold,
)
from dcaf.domains.geometry.generalization import (
    GeneralizationResult,
    compute_generalization_ratio,
    compute_generalization,
    compute_generalization_simple,
    is_generalizable,
    is_overfitting,
    get_generalization_summary,
)
from dcaf.domains.geometry.lrs import (
    LRSBreakdown,
    LRSResult,
    power_mean,
    compute_lrs,
    compute_lrs_from_breakdown,
    compute_lrs_batch,
    is_strong_representation,
    is_weak_representation,
    get_lrs_summary,
)
from dcaf.domains.geometry.confidence import (
    GeometryConfidenceResult,
    compute_geometry_confidence,
    compute_geometry_confidence_simple,
    compute_all_geometry_confidences,
    filter_by_geometry_confidence,
    get_geometry_confidence_summary,
    rank_by_geometry_confidence,
    get_component_breakdown,
)
from dcaf.domains.geometry.nonlinear import (
    NonlinearDiagnostics,
    compute_pacmap_on_activations,
    compute_pacmap_on_deltas,
    compute_procrustes_alignment,
    compute_nonlinear_diagnostics,
)
from dcaf.domains.geometry.probing import (
    ProbingResults,
    compute_kernel_lda,
    compute_polynomial_probing,
)

__all__ = [
    # directions (Def 6.2, 6.4-6.5)
    "DirectionDynamics",
    "compute_pooled_covariance",
    "extract_contrastive_direction",
    "extract_contrastive_directions_batch",
    "compute_direction_emergence",
    "compute_direction_rotation",
    "compute_direction_dynamics",
    "aggregate_directions",
    # alignment (Def 6.8-6.10)
    "ClusterMetrics",
    "compute_alignment_matrix",
    "compute_alignment_matrix_indexed",
    "compute_cluster_coherence",
    "compute_cluster_opposition",
    "compute_baseline_orthogonality",
    "compute_cluster_metrics",
    "compute_cluster_metrics_from_directions",
    "get_alignment_summary",
    # confound (Def 6.6-6.7)
    "ConfoundAnalysis",
    "extract_confound_direction",
    "compute_direction_overlap",
    "compute_confound_independence",
    "compute_confound_analysis",
    "is_contaminated",
    "is_clean",
    # predictivity (Def 6.11-6.12)
    "PredictivityResult",
    "compute_auc",
    "compute_predictivity",
    "compute_predictivity_gain",
    "compute_predictivity_gain_batch",
    "normalize_predictivity_gain",
    "compute_predictivity_at_threshold",
    # generalization (Def 6.13)
    "GeneralizationResult",
    "compute_generalization_ratio",
    "compute_generalization",
    "compute_generalization_simple",
    "is_generalizable",
    "is_overfitting",
    "get_generalization_summary",
    # lrs (Def 6.14)
    "LRSBreakdown",
    "LRSResult",
    "power_mean",
    "compute_lrs",
    "compute_lrs_from_breakdown",
    "compute_lrs_batch",
    "is_strong_representation",
    "is_weak_representation",
    "get_lrs_summary",
    # confidence (Def 6.15)
    "GeometryConfidenceResult",
    "compute_geometry_confidence",
    "compute_geometry_confidence_simple",
    "compute_all_geometry_confidences",
    "filter_by_geometry_confidence",
    "get_geometry_confidence_summary",
    "rank_by_geometry_confidence",
    "get_component_breakdown",
    # nonlinear (Def 6.16-6.18)
    "NonlinearDiagnostics",
    "compute_pacmap_on_activations",
    "compute_pacmap_on_deltas",
    "compute_procrustes_alignment",
    "compute_nonlinear_diagnostics",
    # probing (Def 6.19)
    "ProbingResults",
    "compute_kernel_lda",
    "compute_polynomial_probing",
]
