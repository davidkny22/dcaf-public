"""
Phase 4: Multi-Strategy Confirmation Counting (sec:phase-4-cross-validation-counting;
def:multi-strategy-confirmation).

def:multi-strategy-confirmation: discovery_count(k) = |{s ∈ Strategies : k found in a significant
pair/group by s}|. Components in H_solo from Phase 1 count as one strategy
(phase1_individual).

  - discovery_count ≥ 2: High-confidence circuit
  - discovery_count = 1: Re-test in different ablation contexts; validate or
                          mark uncertain

This module also implements cross-validation by testing parameters with
different test functions/prompt sets to verify consistency across evaluation
methods.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dcaf.ablation.methods import ModelStateManager
from dcaf.core.defaults import HIGH_CONSISTENCY, TAU_ABS


@dataclass
class CrossValidationResult:
    """
    Result of cross-validation with multiple test functions.

    Attributes:
        params: Parameters tested
        test_results: {test_name: passed}
        passed_count: Number of tests passed
        total_tests: Total number of tests
        consistency_score: passed_count / total_tests
        details: {test_name: {score, threshold, passed}}
    """
    params: List[str]
    test_results: Dict[str, bool] = field(default_factory=dict)
    passed_count: int = 0
    total_tests: int = 0
    consistency_score: float = 0.0
    details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "test_results": self.test_results,
            "passed_count": self.passed_count,
            "total_tests": self.total_tests,
            "consistency_score": self.consistency_score,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossValidationResult":
        return cls(
            params=data["params"],
            test_results=data.get("test_results", {}),
            passed_count=data.get("passed_count", 0),
            total_tests=data.get("total_tests", 0),
            consistency_score=data.get("consistency_score", 0.0),
            details=data.get("details", {}),
        )


def cross_validate(
    params: List[str],
    model,
    state_manager: ModelStateManager,
    test_functions: Dict[str, Callable[..., float]],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> CrossValidationResult:
    """
    Test parameters with different test functions/prompt sets.

    Args:
        params: Parameters to test
        model: Model to test
        state_manager: ModelStateManager for ablation
        test_functions: {test_name: test_fn}
        impact_threshold: Minimum impact to pass
        test_kwargs: {test_name: kwargs} for each test function

    Returns:
        CrossValidationResult with consistency score
    """
    if test_kwargs is None:
        test_kwargs = {}

    test_results = {}
    details = {}

    for test_name, test_fn in test_functions.items():
        kwargs = test_kwargs.get(test_name, {})

        # Get baseline
        state_manager.reset_to_safety()
        baseline = test_fn(model, **kwargs)

        # Get ablated
        with state_manager.temporary_ablation(params):
            ablated = test_fn(model, **kwargs)

        impact = abs(ablated - baseline)
        passed = impact >= impact_threshold

        test_results[test_name] = passed
        details[test_name] = {
            "baseline": baseline,
            "ablated": ablated,
            "impact": impact,
            "threshold": impact_threshold,
            "passed": passed,
        }

    passed_count = sum(test_results.values())
    total_tests = len(test_functions)
    consistency_score = passed_count / total_tests if total_tests > 0 else 0.0

    return CrossValidationResult(
        params=params,
        test_results=test_results,
        passed_count=passed_count,
        total_tests=total_tests,
        consistency_score=consistency_score,
        details=details,
    )


def cross_validate_batch(
    param_groups: List[List[str]],
    model,
    state_manager: ModelStateManager,
    test_functions: Dict[str, Callable[..., float]],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[CrossValidationResult]:
    """
    Cross-validate multiple parameter groups.

    Args:
        param_groups: List of parameter groups
        model: Model to test
        state_manager: ModelStateManager
        test_functions: {test_name: test_fn}
        impact_threshold: Minimum impact
        test_kwargs: {test_name: kwargs}

    Returns:
        List of CrossValidationResult
    """
    results = []
    for params in param_groups:
        result = cross_validate(
            params, model, state_manager, test_functions,
            impact_threshold=impact_threshold,
            test_kwargs=test_kwargs,
        )
        results.append(result)
    return results


def filter_consistent(
    results: List[CrossValidationResult],
    min_consistency: float = HIGH_CONSISTENCY,
) -> List[CrossValidationResult]:
    """
    Filter to only results with high consistency.

    Args:
        results: Cross-validation results
        min_consistency: Minimum consistency score to keep

    Returns:
        Filtered list
    """
    return [r for r in results if r.consistency_score >= min_consistency]


__all__ = [
    "CrossValidationResult",
    "cross_validate",
    "cross_validate_batch",
    "filter_consistent",
]
