"""
Phase 5: Triple detection for GATE patterns (§11, Def 11.14-11.15).

Def 11.14 (Approach 1): For each Strategy B pair with low behavioral ablation
impact, add third members: components with discovery_count ≥ 2 from Phase 4,
and high-confidence components (top of H_cand by C^(k)).

Def 11.15 (Approach 2): Components with no individual or pair effect; combine
with top 5 H_solo components as triples. Also form triples among no-effect
components with similar opposition profiles (|opp_degree delta| < 0.1).

Total triple budget B_triple (default 100), prioritizing high discovery-count
components.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from itertools import combinations

from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.superadditivity import (
    InteractionRequirement,
    InteractionType,
    classify_interaction_requirement,
)
from dcaf.core.defaults import EPSILON_TRI, TAU_ABS


@dataclass
class TriplesResult:
    """
    Result of triple testing for GATE detection.

    Attributes:
        params: The three parameters tested
        individual_impacts: {param: impact}
        pair_impacts: {(p1, p2): impact}
        triple_impact: Impact when all three ablated
        is_gate: True if GATE pattern detected (3+ params required)
        interaction_requirement: SOLO, PAIR, or GATE
        interaction_type: SYNERGISTIC, REDUNDANT, etc.
    """
    params: List[str]
    individual_impacts: Dict[str, float] = field(default_factory=dict)
    pair_impacts: Dict[str, float] = field(default_factory=dict)
    triple_impact: float = 0.0
    is_gate: bool = False
    interaction_requirement: str = "SOLO"
    interaction_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "individual_impacts": self.individual_impacts,
            "pair_impacts": self.pair_impacts,
            "triple_impact": self.triple_impact,
            "is_gate": self.is_gate,
            "interaction_requirement": self.interaction_requirement,
            "interaction_type": self.interaction_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriplesResult":
        return cls(
            params=data["params"],
            individual_impacts=data.get("individual_impacts", {}),
            pair_impacts=data.get("pair_impacts", {}),
            triple_impact=data.get("triple_impact", 0.0),
            is_gate=data.get("is_gate", False),
            interaction_requirement=data.get("interaction_requirement", "SOLO"),
            interaction_type=data.get("interaction_type"),
        )


def test_triple(
    params: List[str],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    impact_threshold: float = TAU_ABS,
    epsilon: float = EPSILON_TRI,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> TriplesResult:
    """
    Test a triple of parameters for GATE pattern.

    GATE: No individual or pair shows significant impact, but triple does.

    Args:
        params: Exactly 3 parameter names
        model: Model to test
        state_manager: ModelStateManager for ablation
        test_fn: Impact measurement function
        impact_threshold: Minimum impact for significance
        epsilon: Tolerance for interaction classification
        test_kwargs: Optional kwargs for test function

    Returns:
        TriplesResult with GATE detection
    """
    if test_kwargs is None:
        test_kwargs = {}

    if len(params) != 3:
        raise ValueError(f"test_triple requires exactly 3 params, got {len(params)}")

    # Get baseline
    state_manager.reset_to_safety()
    baseline = test_fn(model, **test_kwargs)

    # Test individuals
    individual_impacts = {}
    for p in params:
        with state_manager.temporary_ablation([p]):
            ablated = test_fn(model, **test_kwargs)
        individual_impacts[p] = abs(ablated - baseline)

    # Test pairs
    pair_impacts = {}
    for p1, p2 in combinations(params, 2):
        with state_manager.temporary_ablation([p1, p2]):
            ablated = test_fn(model, **test_kwargs)
        pair_impacts[f"{p1}+{p2}"] = abs(ablated - baseline)

    # Test triple
    with state_manager.temporary_ablation(params):
        ablated = test_fn(model, **test_kwargs)
    triple_impact = abs(ablated - baseline)

    # Determine if GATE pattern
    any_individual_significant = any(i >= impact_threshold for i in individual_impacts.values())
    any_pair_significant = any(i >= impact_threshold for i in pair_impacts.values())
    triple_significant = triple_impact >= impact_threshold

    is_gate = (
        not any_individual_significant and
        not any_pair_significant and
        triple_significant
    )

    # Classify interaction requirement
    interaction_req = classify_interaction_requirement(
        individual_impacts, triple_impact, impact_threshold
    )

    # Classify interaction type (superadditivity)
    sum_individuals = sum(individual_impacts.values())
    if triple_impact > sum_individuals + epsilon:
        interaction_type = InteractionType.SYNERGISTIC.value
    elif triple_impact < sum_individuals - epsilon:
        interaction_type = InteractionType.SUBADDITIVE.value
    else:
        interaction_type = InteractionType.ADDITIVE.value

    return TriplesResult(
        params=params,
        individual_impacts=individual_impacts,
        pair_impacts=pair_impacts,
        triple_impact=triple_impact,
        is_gate=is_gate,
        interaction_requirement=interaction_req.value,
        interaction_type=interaction_type,
    )


def test_triples_batch(
    triple_candidates: List[List[str]],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> List[TriplesResult]:
    """
    Test multiple triples for GATE patterns.

    Args:
        triple_candidates: List of [p1, p2, p3] parameter groups
        model: Model to test
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        impact_threshold: Minimum impact
        test_kwargs: Optional kwargs for test function

    Returns:
        List of TriplesResult
    """
    results = []
    for params in triple_candidates:
        if len(params) == 3:
            result = test_triple(
                params, model, state_manager, test_fn,
                impact_threshold=impact_threshold,
                test_kwargs=test_kwargs,
            )
            results.append(result)
    return results


def filter_gates(results: List[TriplesResult]) -> List[TriplesResult]:
    """Filter to only GATE patterns."""
    return [r for r in results if r.is_gate]


def filter_synergistic_triples(results: List[TriplesResult]) -> List[TriplesResult]:
    """Filter to synergistic triple interactions."""
    return [r for r in results if r.interaction_type == InteractionType.SYNERGISTIC.value]


__all__ = [
    "TriplesResult",
    "test_triple",
    "test_triples_batch",
    "filter_gates",
    "filter_synergistic_triples",
]
