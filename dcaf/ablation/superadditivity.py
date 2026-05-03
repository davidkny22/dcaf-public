"""
Interaction type classification and superadditivity testing
(sec:superadditivity-and-interaction-classification; def:interaction-type-three).

def:interaction-type-three: Interaction types for PAIR and GATE components:
    SYNERGISTIC:  I(k1,k2) > I(k1) + I(k2) + ε_syn
    ADDITIVE:     |I(k1,k2) - (I(k1) + I(k2))| ≤ ε_syn
    REDUNDANT:    I(k1,k2) ≈ max(I(k1), I(k2))

where ε_syn is the synergy detection threshold (default 0.05, EPSILON_TRI).

InteractionRequirement (from def:interaction-requirement): SOLO / PAIR / GATE — how many
components are required for the behavioral effect (separate from how they
combine when they do).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dcaf.ablation.methods import ModelStateManager
from dcaf.core.defaults import EPSILON_TRI, SYNERGY_EPSILON, TAU_ABS


class InteractionType(Enum):
    """Type of interaction between parameters/components (how they combine)."""
    SYNERGISTIC = "synergistic"  # Combined > sum of individuals
    REDUNDANT = "redundant"      # Combined ≈ max of individuals
    ADDITIVE = "additive"        # Combined ≈ sum of individuals
    SUBADDITIVE = "subadditive"  # Combined < sum of individuals
    SOLO = "solo"                # Single parameter, no interaction


class InteractionRequirement(Enum):
    """How many parameters are needed for the behavioral effect."""
    SOLO = "solo"   # Single parameter is sufficient
    PAIR = "pair"   # Exactly 2 parameters required together
    GATE = "gate"   # 3+ parameters required (gating structure)


@dataclass
class SuperadditivityResult:
    """
    Result of superadditivity testing.

    Attributes:
        params: Parameters tested
        individual_impacts: {param: impact_value}
        combined_impact: Impact when all params ablated together
        superadditive: Whether combined > sum(individual) + epsilon
        interaction_strength: combined - sum(individual)
        interaction_type: How params combine (SYNERGISTIC/REDUNDANT/etc.)
        interaction_requirement: How many params needed (SOLO/PAIR/GATE)
    """
    params: List[str]
    individual_impacts: Dict[str, float]
    combined_impact: float
    superadditive: bool
    interaction_strength: float
    interaction_type: InteractionType
    interaction_requirement: InteractionRequirement = InteractionRequirement.SOLO
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def sum_individual(self) -> float:
        """Sum of individual impacts."""
        return sum(self.individual_impacts.values())

    @property
    def max_individual(self) -> float:
        """Maximum individual impact."""
        return max(self.individual_impacts.values()) if self.individual_impacts else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "individual_impacts": self.individual_impacts,
            "combined_impact": self.combined_impact,
            "sum_individual": self.sum_individual,
            "max_individual": self.max_individual,
            "superadditive": self.superadditive,
            "interaction_strength": self.interaction_strength,
            "interaction_type": self.interaction_type.value,
            "interaction_requirement": self.interaction_requirement.value,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperadditivityResult":
        return cls(
            params=data["params"],
            individual_impacts=data["individual_impacts"],
            combined_impact=data["combined_impact"],
            superadditive=data["superadditive"],
            interaction_strength=data["interaction_strength"],
            interaction_type=InteractionType(data["interaction_type"]),
            interaction_requirement=InteractionRequirement(
                data.get("interaction_requirement", "solo")
            ),
            details=data.get("details", {}),
        )


def classify_interaction(
    I_combined: float,
    I_individuals: List[float],
    epsilon: float = EPSILON_TRI,
) -> InteractionType:
    """
    Classify interaction type based on impact values.

    SYNERGISTIC: I(p1,p2) > I(p1) + I(p2) + ε
    REDUNDANT: I(p1,p2) ≈ max(I(p1), I(p2))
    ADDITIVE: I(p1,p2) ≈ I(p1) + I(p2)
    SUBADDITIVE: I(p1,p2) < I(p1) + I(p2) - ε

    Args:
        I_combined: Impact when all ablated together
        I_individuals: List of individual impacts
        epsilon: Tolerance for comparisons

    Returns:
        InteractionType
    """
    if len(I_individuals) == 0:
        return InteractionType.SOLO

    if len(I_individuals) == 1:
        return InteractionType.SOLO

    sum_individual = sum(I_individuals)
    max_individual = max(I_individuals)

    # Synergistic: combined significantly exceeds sum
    if I_combined > sum_individual + epsilon:
        return InteractionType.SYNERGISTIC

    # Redundant: combined is close to max (not additive)
    if abs(I_combined - max_individual) < epsilon:
        return InteractionType.REDUNDANT

    # Additive: combined is close to sum
    if abs(I_combined - sum_individual) < epsilon:
        return InteractionType.ADDITIVE

    # Subadditive: combined is less than sum
    if I_combined < sum_individual - epsilon:
        return InteractionType.SUBADDITIVE

    # Default to additive if within tolerance
    return InteractionType.ADDITIVE


def classify_interaction_requirement(
    individual_impacts: Dict[str, float],
    combined_impact: float,
    impact_threshold: float = TAU_ABS,
) -> InteractionRequirement:
    """
    Classify how many parameters are REQUIRED for the behavioral effect.

    This is separate from InteractionType (how params combine):
    - SOLO: At least one individual impact >= threshold (single param sufficient)
    - PAIR: No individual >= threshold, but combined >= threshold, n <= 2
    - GATE: No individual >= threshold, combined >= threshold, n >= 3

    Args:
        individual_impacts: {param: individual_impact_value}
        combined_impact: Impact when all ablated together
        impact_threshold: Minimum impact to consider "effective"

    Returns:
        InteractionRequirement (SOLO, PAIR, or GATE)
    """
    n_params = len(individual_impacts)

    if n_params == 0:
        return InteractionRequirement.SOLO

    # Check if any individual param is sufficient
    any_solo = any(v >= impact_threshold for v in individual_impacts.values())

    if any_solo:
        return InteractionRequirement.SOLO
    elif combined_impact >= impact_threshold:
        # Need multiple params together
        if n_params <= 2:
            return InteractionRequirement.PAIR
        else:
            return InteractionRequirement.GATE
    else:
        # Even combined doesn't meet threshold
        return InteractionRequirement.SOLO


def test_superadditivity(
    params: List[str],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    epsilon: float = SYNERGY_EPSILON,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> SuperadditivityResult:
    """
    Test if combined ablation > sum of individual ablations.

    Args:
        params: List of parameter names to test
        model: Model to ablate
        state_manager: ModelStateManager for ablation
        test_fn: Function that returns impact score (higher = more impact)
        epsilon: Tolerance for superadditivity
        test_kwargs: Optional kwargs for test function

    Returns:
        SuperadditivityResult
    """
    if test_kwargs is None:
        test_kwargs = {}

    if len(params) == 0:
        return SuperadditivityResult(
            params=[],
            individual_impacts={},
            combined_impact=0.0,
            superadditive=False,
            interaction_strength=0.0,
            interaction_type=InteractionType.SOLO,
            details={"error": "No parameters provided"},
        )

    if len(params) == 1:
        # Single parameter - measure its impact
        state_manager.reset_to_safety()
        baseline = test_fn(model, **test_kwargs)

        with state_manager.temporary_ablation(params):
            ablated = test_fn(model, **test_kwargs)

        impact = abs(ablated - baseline)

        # Single param: classify requirement
        req = classify_interaction_requirement(
            {params[0]: impact}, impact, epsilon
        )
        return SuperadditivityResult(
            params=params,
            individual_impacts={params[0]: impact},
            combined_impact=impact,
            superadditive=False,
            interaction_strength=0.0,
            interaction_type=InteractionType.SOLO,
            interaction_requirement=req,
        )

    # Test individual impacts
    state_manager.reset_to_safety()
    baseline = test_fn(model, **test_kwargs)

    individual_impacts = {}
    for param in params:
        with state_manager.temporary_ablation([param]):
            ablated = test_fn(model, **test_kwargs)
        individual_impacts[param] = abs(ablated - baseline)

    # Test combined impact
    state_manager.reset_to_safety()
    with state_manager.temporary_ablation(params):
        combined_ablated = test_fn(model, **test_kwargs)
    combined_impact = abs(combined_ablated - baseline)

    # Calculate interaction strength
    sum_individual = sum(individual_impacts.values())
    interaction_strength = combined_impact - sum_individual

    # Classify interaction
    interaction_type = classify_interaction(
        combined_impact,
        list(individual_impacts.values()),
        epsilon,
    )

    # Determine superadditivity
    superadditive = interaction_type == InteractionType.SYNERGISTIC

    # Classify interaction requirement (how many params needed)
    interaction_requirement = classify_interaction_requirement(
        individual_impacts, combined_impact, epsilon
    )

    return SuperadditivityResult(
        params=params,
        individual_impacts=individual_impacts,
        combined_impact=combined_impact,
        superadditive=superadditive,
        interaction_strength=interaction_strength,
        interaction_type=interaction_type,
        interaction_requirement=interaction_requirement,
        details={
            "baseline": baseline,
            "epsilon": epsilon,
        },
    )


def test_pair_superadditivity(
    param1: str,
    param2: str,
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    epsilon: float = EPSILON_TRI,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> SuperadditivityResult:
    """
    Test superadditivity for a pair of parameters.

    Convenience wrapper for test_superadditivity with two params.

    Args:
        param1: First parameter
        param2: Second parameter
        model: Model to ablate
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        epsilon: Tolerance
        test_kwargs: Optional kwargs for test function

    Returns:
        SuperadditivityResult
    """
    return test_superadditivity(
        params=[param1, param2],
        model=model,
        state_manager=state_manager,
        test_fn=test_fn,
        epsilon=epsilon,
        test_kwargs=test_kwargs,
    )


def batch_test_superadditivity(
    param_groups: List[List[str]],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    epsilon: float = EPSILON_TRI,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> List[SuperadditivityResult]:
    """
    Test superadditivity for multiple parameter groups.

    Args:
        param_groups: List of parameter groups to test
        model: Model to ablate
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        epsilon: Tolerance
        test_kwargs: Optional kwargs for test function

    Returns:
        List of SuperadditivityResults
    """
    results = []

    for params in param_groups:
        result = test_superadditivity(
            params=params,
            model=model,
            state_manager=state_manager,
            test_fn=test_fn,
            epsilon=epsilon,
            test_kwargs=test_kwargs,
        )
        results.append(result)

    return results


def filter_synergistic(
    results: List[SuperadditivityResult],
) -> List[SuperadditivityResult]:
    """Filter to only synergistic interactions."""
    return [r for r in results if r.interaction_type == InteractionType.SYNERGISTIC]


def filter_redundant(
    results: List[SuperadditivityResult],
) -> List[SuperadditivityResult]:
    """Filter to only redundant interactions."""
    return [r for r in results if r.interaction_type == InteractionType.REDUNDANT]


def rank_by_interaction_strength(
    results: List[SuperadditivityResult],
) -> List[SuperadditivityResult]:
    """Rank results by interaction strength descending."""
    return sorted(results, key=lambda r: r.interaction_strength, reverse=True)


def get_superadditivity_summary(
    results: List[SuperadditivityResult],
) -> Dict[str, Any]:
    """
    Summary statistics for superadditivity results.

    Args:
        results: List of SuperadditivityResult

    Returns:
        Summary dict
    """
    if not results:
        return {"count": 0}

    type_counts = {}
    for result in results:
        t = result.interaction_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    strengths = [r.interaction_strength for r in results]

    return {
        "count": len(results),
        "type_counts": type_counts,
        "synergistic_count": type_counts.get("synergistic", 0),
        "redundant_count": type_counts.get("redundant", 0),
        "mean_interaction_strength": sum(strengths) / len(strengths),
        "max_interaction_strength": max(strengths),
        "min_interaction_strength": min(strengths),
    }


__all__ = [
    "InteractionType",
    "InteractionRequirement",
    "SuperadditivityResult",
    "classify_interaction",
    "classify_interaction_requirement",
    "test_superadditivity",
    "test_pair_superadditivity",
    "batch_test_superadditivity",
    "filter_synergistic",
    "filter_redundant",
    "rank_by_interaction_strength",
    "get_superadditivity_summary",
]
