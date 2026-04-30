"""
Probe-specific impact measurement (§11, Def 11.3, Def 11.5).

Def 11.3: Probe-specific impact for confirmed component k:

    I_π^(k) = |score_intact^π - score_ablated^π| / (|score_intact^π| + ε)

where ε prevents division by zero.

Def 11.5: Individual ablation — ablate each k ∈ H_cand, classify behavioral
relevance, measure I_π^(k) for all probe types if behaviorally relevant.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

from dcaf.core.defaults import CLASSIFICATION_THRESHOLD

PROBE_CLASSIFICATION_IDS = {
    "recognition": "I_detect",
    "free_generation": "I_decide",
    "teacher_forcing": "I_eval",
}


@dataclass
class ProbeImpact:
    """
    Impact measurement for a single probe type.

    Attributes:
        probe_type: Type of probe ("recognition", "free_generation", "teacher_forcing")
        impact: Absolute impact value I_π^(k) = |score_pre - score_post|
        pre_score: Score before ablation
        post_score: Score after ablation
    """
    probe_type: str
    impact: float
    pre_score: float
    post_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_type": self.probe_type,
            "impact": self.impact,
            "pre_score": self.pre_score,
            "post_score": self.post_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeImpact":
        return cls(
            probe_type=data["probe_type"],
            impact=data["impact"],
            pre_score=data["pre_score"],
            post_score=data["post_score"],
        )


@dataclass
class ComponentImpact:
    """
    Aggregated impact measurements for a component.

    Attributes:
        component: Component ID (e.g., "L10_MLP")
        I_detect: Impact on detection/recognition probe
        I_decide: Impact on decision/free generation probe
        I_eval: Impact on evaluation/teacher forcing probe
        probe_impacts: Detailed per-probe impacts
    """
    component: str
    I_detect: float
    I_decide: float
    I_eval: float
    probe_impacts: Dict[str, ProbeImpact] = field(default_factory=dict)

    @property
    def max_impact(self) -> float:
        """Maximum impact across all probes."""
        return max(self.I_detect, self.I_decide, self.I_eval)

    @property
    def total_impact(self) -> float:
        """Sum of all probe impacts."""
        return self.I_detect + self.I_decide + self.I_eval

    def impacts_above_threshold(self, tau: float = CLASSIFICATION_THRESHOLD) -> int:
        """Count of probes with impact above threshold.

        Args:
            tau: Minimum impact value to count as "above threshold".
                 Defaults to CLASSIFICATION_THRESHOLD.

        Returns:
            Number of probes (0-3) with impact exceeding tau.
        """
        return sum(1 for i in [self.I_detect, self.I_decide, self.I_eval] if i > tau)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "I_detect": self.I_detect,
            "I_decide": self.I_decide,
            "I_eval": self.I_eval,
            "max_impact": self.max_impact,
            "total_impact": self.total_impact,
            "probe_impacts": {k: v.to_dict() for k, v in self.probe_impacts.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentImpact":
        probe_impacts = {
            k: ProbeImpact.from_dict(v)
            for k, v in data.get("probe_impacts", {}).items()
        }
        return cls(
            component=data["component"],
            I_detect=data["I_detect"],
            I_decide=data["I_decide"],
            I_eval=data["I_eval"],
            probe_impacts=probe_impacts,
        )


def compute_probe_impact(
    score_pre: float,
    score_post: float,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute normalized impact for a single probe (Def 11.3).

    I_π^(k) = |score_intact^π - score_ablated^π| / (|score_intact^π| + ε)

    Args:
        score_pre: Score with model intact (before ablation)
        score_post: Score after ablation
        epsilon: Small constant to prevent division by zero

    Returns:
        Normalized impact in [0, ∞) (typically [0, 1] for rate-type scores)
    """
    return abs(score_pre - score_post) / (abs(score_pre) + epsilon)


def compute_probe_impact_detailed(
    probe_type: str,
    score_pre: float,
    score_post: float,
    epsilon: float = 1e-8,
) -> ProbeImpact:
    """
    Compute detailed impact for a single probe (Def 11.3).

    Args:
        probe_type: Type of probe
        score_pre: Score with model intact
        score_post: Score after ablation
        epsilon: Small constant to prevent division by zero

    Returns:
        ProbeImpact with full details
    """
    return ProbeImpact(
        probe_type=probe_type,
        impact=compute_probe_impact(score_pre, score_post, epsilon),
        pre_score=score_pre,
        post_score=score_post,
    )


def compute_component_impact(
    component: str,
    scores_pre: Dict[str, float],
    scores_post: Dict[str, float],
    epsilon: float = 1e-8,
) -> ComponentImpact:
    """
    Compute impact across all probe types for a component.

    Args:
        component: Component ID
        scores_pre: {probe_type: score} with model intact
        scores_post: {probe_type: score} after ablation
        epsilon: Small constant to prevent division by zero (Def 11.3)

    Returns:
        ComponentImpact with all probe impacts
    """
    probe_impacts = {}

    I_detect = 0.0
    I_decide = 0.0
    I_eval = 0.0

    for probe_type in scores_pre.keys():
        if probe_type not in scores_post:
            continue

        probe_impact = compute_probe_impact_detailed(
            probe_type,
            scores_pre[probe_type],
            scores_post[probe_type],
            epsilon=epsilon,
        )
        probe_impacts[probe_type] = probe_impact

        # Map to standard impact names
        impact_id = PROBE_CLASSIFICATION_IDS.get(probe_type)
        if impact_id == "I_detect":
            I_detect = probe_impact.impact
        elif impact_id == "I_decide":
            I_decide = probe_impact.impact
        elif impact_id == "I_eval":
            I_eval = probe_impact.impact

    return ComponentImpact(
        component=component,
        I_detect=I_detect,
        I_decide=I_decide,
        I_eval=I_eval,
        probe_impacts=probe_impacts,
    )


def compute_component_impact_from_ablation(
    component: str,
    model,
    state_manager,
    probes: Dict[str, Callable[..., float]],
    probe_kwargs: Optional[Dict[str, Any]] = None,
) -> ComponentImpact:
    """
    Compute component impact by running ablation.

    Args:
        component: Component ID to ablate
        model: Model to test
        state_manager: ModelStateManager for ablation
        probes: {probe_type: callable} where callable returns a score
        probe_kwargs: Optional kwargs to pass to probe functions

    Returns:
        ComponentImpact with measured impacts
    """
    if probe_kwargs is None:
        probe_kwargs = {}

    # Get pre-ablation scores
    state_manager.reset_to_safety()
    scores_pre = {
        probe_type: probe_fn(model, **probe_kwargs)
        for probe_type, probe_fn in probes.items()
    }

    # Get post-ablation scores
    component_params = _get_component_params(component, state_manager)
    with state_manager.temporary_ablation(component_params):
        scores_post = {
            probe_type: probe_fn(model, **probe_kwargs)
            for probe_type, probe_fn in probes.items()
        }

    return compute_component_impact(component, scores_pre, scores_post)


def _get_component_params(component: str, state_manager) -> List[str]:
    """Get all parameters belonging to a component.

    Delegates to the canonical implementation in dcaf.arch.transformer.
    """
    from dcaf.arch.transformer import get_component_params
    return get_component_params(component, state_manager.get_delta_params())


def aggregate_impacts(
    impacts: Dict[str, ComponentImpact],
) -> Dict[str, float]:
    """
    Aggregate impacts for ranking.

    Args:
        impacts: {component: ComponentImpact}

    Returns:
        {component: aggregated_score} sorted descending
    """
    aggregated = {
        component: impact.total_impact
        for component, impact in impacts.items()
    }
    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def rank_by_impact(
    impacts: Dict[str, ComponentImpact],
    probe_type: Optional[str] = None,
) -> List[tuple]:
    """
    Rank components by impact.

    Args:
        impacts: {component: ComponentImpact}
        probe_type: Optional specific probe to rank by (None = total)

    Returns:
        [(component, impact_value), ...] sorted descending
    """
    if probe_type is None:
        return sorted(
            [(c, i.total_impact) for c, i in impacts.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    # Rank by specific probe
    impact_key = PROBE_CLASSIFICATION_IDS.get(probe_type, probe_type)

    return sorted(
        [(c, getattr(i, impact_key, 0.0)) for c, i in impacts.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def get_impact_summary(
    impacts: Dict[str, ComponentImpact],
) -> Dict[str, Any]:
    """
    Summary statistics for component impacts.

    Args:
        impacts: {component: ComponentImpact}

    Returns:
        Summary dict
    """
    if not impacts:
        return {"count": 0}

    all_detect = [i.I_detect for i in impacts.values()]
    all_decide = [i.I_decide for i in impacts.values()]
    all_eval = [i.I_eval for i in impacts.values()]

    return {
        "count": len(impacts),
        "mean_I_detect": sum(all_detect) / len(all_detect),
        "mean_I_decide": sum(all_decide) / len(all_decide),
        "mean_I_eval": sum(all_eval) / len(all_eval),
        "max_I_detect": max(all_detect),
        "max_I_decide": max(all_decide),
        "max_I_eval": max(all_eval),
    }


__all__ = [
    "PROBE_CLASSIFICATION_IDS",
    "ProbeImpact",
    "ComponentImpact",
    "compute_probe_impact",
    "compute_probe_impact_detailed",
    "compute_component_impact",
    "compute_component_impact_from_ablation",
    "aggregate_impacts",
    "rank_by_impact",
    "get_impact_summary",
]
