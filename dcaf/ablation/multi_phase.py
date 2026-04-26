"""
Multi-phase ablation aggregator (§11, Phases 3-6).

Aggregates results from Phases 3-6 into a single result object:
  Phase 3: Refinement (leave-one-out minimal subset)
  Phase 4: Cross-validation counting (multi-strategy discovery counts)
  Phase 5: Triple/GATE detection
  Phase 6: Orphan retesting
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

from dcaf.ablation.refinement import RefinementResult
from dcaf.ablation.counting import CrossValidationResult
from dcaf.ablation.triples import TriplesResult
from dcaf.ablation.orphans import OrphanTestResult
from dcaf.ablation.superadditivity import InteractionType


@dataclass
class MultiPhaseAblationResult:
    """
    Aggregated results from all ablation phases.

    Attributes:
        refinement_results: Phase 2 refinement results
        cross_validation_results: Phase 3 cross-validation results
        triples_results: Phase 4 triple/GATE detection results
        orphan_results: Phase 5 orphan testing results
        summary: Aggregated statistics
    """
    refinement_results: List[RefinementResult] = field(default_factory=list)
    cross_validation_results: List[CrossValidationResult] = field(default_factory=list)
    triples_results: List[TriplesResult] = field(default_factory=list)
    orphan_results: List[OrphanTestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        # Refinement stats
        groups_refined = len(self.refinement_results)
        total_original = sum(len(r.original_set) for r in self.refinement_results)
        total_minimal = sum(len(r.minimal_set) for r in self.refinement_results)
        avg_reduction = (
            sum(r.reduction_ratio for r in self.refinement_results) / groups_refined
            if groups_refined > 0 else 1.0
        )

        # Cross-validation stats
        cv_count = len(self.cross_validation_results)
        avg_consistency = (
            sum(r.consistency_score for r in self.cross_validation_results) / cv_count
            if cv_count > 0 else 0.0
        )

        # Triples stats
        gates_found = sum(1 for r in self.triples_results if r.is_gate)
        synergistic_count = sum(
            1 for r in self.triples_results
            if r.interaction_type == InteractionType.SYNERGISTIC.value
        )

        # Orphan stats
        orphans_tested = len(self.orphan_results)
        orphans_confirmed = sum(1 for r in self.orphan_results if r.is_confirmed)

        self.summary = {
            "refinement": {
                "groups_refined": groups_refined,
                "total_original_params": total_original,
                "total_minimal_params": total_minimal,
                "avg_reduction_ratio": avg_reduction,
            },
            "cross_validation": {
                "groups_tested": cv_count,
                "avg_consistency_score": avg_consistency,
            },
            "triples": {
                "triples_tested": len(self.triples_results),
                "gates_found": gates_found,
                "synergistic_count": synergistic_count,
            },
            "orphans": {
                "orphans_tested": orphans_tested,
                "orphans_confirmed": orphans_confirmed,
                "confirmation_rate": (
                    orphans_confirmed / orphans_tested if orphans_tested > 0 else 0.0
                ),
            },
        }
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        if not self.summary:
            self.compute_summary()

        return {
            "refinement_results": [r.to_dict() for r in self.refinement_results],
            "cross_validation_results": [r.to_dict() for r in self.cross_validation_results],
            "triples_results": [r.to_dict() for r in self.triples_results],
            "orphan_results": [r.to_dict() for r in self.orphan_results],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPhaseAblationResult":
        return cls(
            refinement_results=[
                RefinementResult.from_dict(r)
                for r in data.get("refinement_results", [])
            ],
            cross_validation_results=[
                CrossValidationResult.from_dict(r)
                for r in data.get("cross_validation_results", [])
            ],
            triples_results=[
                TriplesResult.from_dict(r)
                for r in data.get("triples_results", [])
            ],
            orphan_results=[
                OrphanTestResult.from_dict(r)
                for r in data.get("orphan_results", [])
            ],
            summary=data.get("summary", {}),
        )


__all__ = [
    "MultiPhaseAblationResult",
]
