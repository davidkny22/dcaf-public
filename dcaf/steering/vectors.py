"""Steering vector optimization (sec:steering).

This module re-exports the real implementation from dcaf.circuit.steering.
The canonical implementation lives in dcaf.circuit.steering and is documented
there in full. This module exists for backward-compatible import paths.
"""

from dcaf.circuit.steering import (
    SteeringAlignment,
    SteeringAnalysis,
    SteeringVector,
    compute_bidirectional_effectiveness,
    compute_cosine_similarity,
    compute_full_steering_analysis,
    compute_steering_alignment,
    compute_steering_effectiveness,
    get_defensive_vectors,
    get_steering_summary,
    optimize_bidirectional_steering,
    optimize_steering_vector,
    rank_by_effectiveness,
)

__all__ = [
    "SteeringVector",
    "SteeringAlignment",
    "SteeringAnalysis",
    "compute_cosine_similarity",
    "optimize_steering_vector",
    "optimize_bidirectional_steering",
    "compute_steering_effectiveness",
    "compute_bidirectional_effectiveness",
    "compute_steering_alignment",
    "get_defensive_vectors",
    "compute_full_steering_analysis",
    "rank_by_effectiveness",
    "get_steering_summary",
]
