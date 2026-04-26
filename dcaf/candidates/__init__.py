"""Candidate set construction and filtering (§8, Def 8.4-8.5)."""

from dcaf.candidates.candidate_set import (
    CandidateStatus,
    CandidateInfo,
    CandidateSet,
    create_discovery_set,
    create_multi_path_discovery_set,
    create_validated_set,
    create_confirmed_set,
)
from dcaf.candidates.ranking import (
    RankingMethod,
    RankedCandidate,
    compute_combined_score,
    rank_candidates,
    rank_components,
    get_ranking_summary,
)
from dcaf.candidates.weight_filter import (
    filter_by_weight_confidence,
    filter_by_weight_percentile,
    filter_by_weight_top_k,
    rank_by_weight_confidence,
    compute_weight_statistics,
)
from dcaf.candidates.activation_filter import (
    filter_by_activation_confidence,
    filter_components_by_activation,
    rank_by_activation_confidence,
    get_activation_confidence_for_params,
)
from dcaf.candidates.geometry_filter import (
    filter_by_geometry_confidence,
    filter_components_by_geometry,
    rank_by_geometry_confidence,
    get_geometry_confidence_for_params,
    filter_by_lrs,
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
