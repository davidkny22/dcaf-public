"""
Phase 6: Orphan analysis (§11, Def 11.16-11.18).

Def 11.16: O = {k ∈ H_cand : k not confirmed by any Phase 1-5 configuration}.
Components that passed all three domain confidence filters (C_W, C_A, C_G)
but showed no individual effect, no significant pair, and no significant triple.

Def 11.17: High-confidence orphans O_high = {k ∈ O : C^(k) > τ_orphan}
(default τ_orphan = 0.6). Re-tested paired with every H_solo component
and with other O_high orphans. Confirmed orphans are promoted to PAIR.

Def 11.18: Low-confidence orphans O_low = {k ∈ O : C^(k) ≤ τ_orphan}.
Flagged for optional exhaustive pairwise testing when budget allows.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

from dcaf.ablation.methods import ModelStateManager
from dcaf.core.defaults import TAU_ABS, TAU_ORPHAN


@dataclass
class OrphanTestResult:
    """
    Result of orphan parameter testing.

    Attributes:
        component: Component ID
        params: Parameters in the component
        confidence: Original unified confidence
        ablation_impact: Measured ablation impact
        is_confirmed: True if ablation shows behavioral relevance
        status: "confirmed", "false_positive", "below_threshold", or "pending"
    """
    component: str
    params: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ablation_impact: float = 0.0
    is_confirmed: bool = False
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "params": self.params,
            "confidence": self.confidence,
            "ablation_impact": self.ablation_impact,
            "is_confirmed": self.is_confirmed,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrphanTestResult":
        return cls(
            component=data["component"],
            params=data.get("params", []),
            confidence=data.get("confidence", 0.0),
            ablation_impact=data.get("ablation_impact", 0.0),
            is_confirmed=data.get("is_confirmed", False),
            status=data.get("status", "pending"),
        )


def test_orphan(
    component: str,
    params: List[str],
    confidence: float,
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> OrphanTestResult:
    """
    Test a high-confidence orphan component.

    Orphans: Components that passed confidence threshold but weren't
    discovered by any Phase 1 strategy.

    Args:
        component: Component ID
        params: Parameters in the component
        confidence: Unified confidence score
        model: Model to test
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        impact_threshold: Minimum impact for confirmation
        test_kwargs: Optional kwargs for test function

    Returns:
        OrphanTestResult
    """
    if test_kwargs is None:
        test_kwargs = {}

    if not params:
        return OrphanTestResult(
            component=component,
            params=params,
            confidence=confidence,
            status="false_positive",
        )

    # Get baseline
    state_manager.reset_to_safety()
    baseline = test_fn(model, **test_kwargs)

    # Ablate component
    with state_manager.temporary_ablation(params):
        ablated = test_fn(model, **test_kwargs)

    impact = abs(ablated - baseline)
    is_confirmed = impact >= impact_threshold

    return OrphanTestResult(
        component=component,
        params=params,
        confidence=confidence,
        ablation_impact=impact,
        is_confirmed=is_confirmed,
        status="confirmed" if is_confirmed else "false_positive",
    )


def test_orphans_batch(
    orphan_components: List[Dict[str, Any]],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    confidence_threshold: float = TAU_ORPHAN,
    impact_threshold: float = TAU_ABS,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> List[OrphanTestResult]:
    """
    Test multiple orphan components.

    Args:
        orphan_components: [{component, params, confidence}, ...]
        model: Model to test
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        confidence_threshold: Minimum confidence for testing
        impact_threshold: Minimum impact for confirmation
        test_kwargs: Optional kwargs for test function

    Returns:
        List of OrphanTestResult
    """
    results = []

    for orphan in orphan_components:
        confidence = orphan.get("confidence", 0.0)

        # Only test high-confidence orphans
        if confidence < confidence_threshold:
            results.append(OrphanTestResult(
                component=orphan["component"],
                params=orphan.get("params", []),
                confidence=confidence,
                status="below_threshold",
            ))
            continue

        result = test_orphan(
            component=orphan["component"],
            params=orphan.get("params", []),
            confidence=confidence,
            model=model,
            state_manager=state_manager,
            test_fn=test_fn,
            impact_threshold=impact_threshold,
            test_kwargs=test_kwargs,
        )
        results.append(result)

    return results


def filter_confirmed_orphans(results: List[OrphanTestResult]) -> List[OrphanTestResult]:
    """Filter to confirmed orphans only."""
    return [r for r in results if r.is_confirmed]


def get_orphan_summary(results: List[OrphanTestResult]) -> Dict[str, Any]:
    """
    Summary statistics for orphan testing.

    Args:
        results: Orphan test results

    Returns:
        Summary dict
    """
    if not results:
        return {"count": 0}

    tested = [r for r in results if r.status not in ["below_threshold", "pending"]]
    confirmed = [r for r in results if r.is_confirmed]
    below_threshold = [r for r in results if r.status == "below_threshold"]

    return {
        "total": len(results),
        "tested": len(tested),
        "confirmed": len(confirmed),
        "false_positive": len(tested) - len(confirmed),
        "below_threshold": len(below_threshold),
        "confirmation_rate": len(confirmed) / len(tested) if tested else 0.0,
        "avg_confidence": (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        ),
        "avg_impact_confirmed": (
            sum(r.ablation_impact for r in confirmed) / len(confirmed) if confirmed else 0.0
        ),
    }


__all__ = [
    "OrphanTestResult",
    "test_orphan",
    "test_orphans_batch",
    "filter_confirmed_orphans",
    "get_orphan_summary",
]
