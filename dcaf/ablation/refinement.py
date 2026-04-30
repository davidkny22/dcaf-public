"""
Phase 3: Refinement (§11, Def 11.8 refinement rules).

Targeted follow-up based on Phase 1 and Phase 2 results:
  1. Solo × Solo pairs: test pairwise among H_solo components
  2. No-solo pair members: test H_solo x H_not-solo components
  3. Hessian pairs with low ablation: flag for triple testing in Phase 5
  4. Minimal sufficient subsets from significant Strategy D clusters

Implements leave-one-out elimination to find the minimal critical subset
of parameters for a confirmed interaction.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

from dcaf.ablation.methods import ModelStateManager
from dcaf.core.defaults import TAU_ABS


@dataclass
class RefinementResult:
    """
    Result of leave-one-out refinement.

    Attributes:
        original_set: Original parameter set
        minimal_set: Minimal critical subset
        removed: Parameters that could be removed without losing effect
        reduction_ratio: |minimal| / |original|
        individual_contributions: {param: impact_when_removed}
    """
    original_set: List[str]
    minimal_set: List[str]
    removed: List[str] = field(default_factory=list)
    reduction_ratio: float = 1.0
    individual_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_set": self.original_set,
            "minimal_set": self.minimal_set,
            "removed": self.removed,
            "reduction_ratio": self.reduction_ratio,
            "individual_contributions": self.individual_contributions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinementResult":
        return cls(
            original_set=data["original_set"],
            minimal_set=data["minimal_set"],
            removed=data.get("removed", []),
            reduction_ratio=data.get("reduction_ratio", 1.0),
            individual_contributions=data.get("individual_contributions", {}),
        )


def refine_group(
    params: List[str],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> RefinementResult:
    """
    Leave-one-out elimination to find minimal critical subset.

    For each parameter, test if removing it from the group significantly
    reduces the combined ablation impact. Keep only essential parameters.

    Args:
        params: List of parameter names in the group
        model: Model to test
        state_manager: ModelStateManager for ablation
        test_fn: Function returning impact score (higher = more impact)
        impact_threshold: Minimum impact to consider significant
        test_kwargs: Optional kwargs for test function

    Returns:
        RefinementResult with minimal critical subset
    """
    if test_kwargs is None:
        test_kwargs = {}

    if len(params) <= 1:
        return RefinementResult(
            original_set=params,
            minimal_set=params,
            reduction_ratio=1.0,
        )

    # Get baseline (all params ablated)
    state_manager.reset_to_safety()
    baseline_score = test_fn(model, **test_kwargs)

    with state_manager.temporary_ablation(params):
        full_ablated_score = test_fn(model, **test_kwargs)
    full_impact = abs(full_ablated_score - baseline_score)

    # Leave-one-out: test each param's contribution
    contributions = {}
    for param in params:
        remaining = [p for p in params if p != param]
        if not remaining:
            contributions[param] = full_impact
            continue

        state_manager.reset_to_safety()
        with state_manager.temporary_ablation(remaining):
            partial_score = test_fn(model, **test_kwargs)
        partial_impact = abs(partial_score - baseline_score)

        # Contribution = how much impact is lost when this param is kept (not ablated)
        contributions[param] = full_impact - partial_impact

    # Find minimal set: keep params whose removal significantly reduces impact
    minimal_set = []
    removed = []

    for param in params:
        # If contribution is significant, keep it
        if contributions[param] >= impact_threshold * full_impact:
            minimal_set.append(param)
        else:
            removed.append(param)

    # Ensure at least one param remains
    if not minimal_set and params:
        # Keep the highest contributor
        best_param = max(contributions.items(), key=lambda x: x[1])[0]
        minimal_set = [best_param]
        removed = [p for p in params if p != best_param]

    return RefinementResult(
        original_set=params,
        minimal_set=minimal_set,
        removed=removed,
        reduction_ratio=len(minimal_set) / len(params) if params else 1.0,
        individual_contributions=contributions,
    )


def refine_groups_batch(
    param_groups: List[List[str]],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> List[RefinementResult]:
    """
    Refine multiple parameter groups.

    Args:
        param_groups: List of parameter groups to refine
        model: Model to test
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        impact_threshold: Minimum impact
        test_kwargs: Optional kwargs for test function

    Returns:
        List of RefinementResult
    """
    results = []
    for params in param_groups:
        result = refine_group(
            params, model, state_manager, test_fn,
            impact_threshold=impact_threshold,
            test_kwargs=test_kwargs,
        )
        results.append(result)
    return results


__all__ = [
    "RefinementResult",
    "refine_group",
    "refine_groups_batch",
]
