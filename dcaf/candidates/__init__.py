"""Candidate set construction and filtering (sec:candidate-set-construction)."""

from dcaf.candidates.activation_filter import (
    filter_by_activation_confidence,
    filter_components_by_activation,
    get_activation_confidence_for_params,
    rank_by_activation_confidence,
)
from dcaf.candidates.candidate_set import (
    CandidateInfo,
    CandidateSet,
    CandidateStatus,
    create_confirmed_set,
    create_discovery_set,
    create_multi_path_discovery_set,
    create_validated_set,
)
from dcaf.candidates.geometry_filter import (
    filter_by_geometry_confidence,
    filter_by_lrs,
    filter_components_by_geometry,
    get_geometry_confidence_for_params,
    rank_by_geometry_confidence,
)
from dcaf.candidates.ranking import (
    RankedCandidate,
    RankingMethod,
    compute_combined_score,
    get_ranking_summary,
    rank_candidates,
    rank_components,
)
from dcaf.candidates.weight_filter import (
    compute_weight_statistics,
    filter_by_weight_confidence,
    filter_by_weight_percentile,
    filter_by_weight_top_k,
    rank_by_weight_confidence,
)

__all__ = [
    # Candidate set data structures
    "CandidateStatus",
    "CandidateInfo",
    "CandidateSet",
    "create_discovery_set",
    "create_multi_path_discovery_set",
    "create_validated_set",
    "create_confirmed_set",
    # Ranking
    "RankingMethod",
    "RankedCandidate",
    "compute_combined_score",
    "rank_candidates",
    "rank_components",
    "get_ranking_summary",
    # Weight filtering
    "filter_by_weight_confidence",
    "filter_by_weight_percentile",
    "filter_by_weight_top_k",
    "rank_by_weight_confidence",
    "compute_weight_statistics",
    # Activation filtering
    "filter_by_activation_confidence",
    "filter_components_by_activation",
    "rank_by_activation_confidence",
    "get_activation_confidence_for_params",
    # Geometry filtering
    "filter_by_geometry_confidence",
    "filter_components_by_geometry",
    "rank_by_geometry_confidence",
    "get_geometry_confidence_for_params",
    "filter_by_lrs",
]
