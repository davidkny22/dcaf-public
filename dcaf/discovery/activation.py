"""
Activation-based parameter discovery H_A
(sec:activation-discovery; def:activation-based-discovery-set).

Identifies leverage points: parameters where small weight changes
produce large activation effects.

Two-stage process (def:component-screening; def:projection-level-activation-score;
def:activation-based-discovery-set):
  Stage 1: K_flagged = {k : m_agg(k) >= Phi_{tau_comp}}  (component screening, tau_comp=70)
  Stage 2: S_A(proj) = (m_agg(k) / max_m) * w_param(proj)
  H_A = {proj : k(proj) in K_flagged AND S_A(proj) >= Phi_{tau_act}}  (tau_act=85)

Note: H_A (discovery) is different from C_A (confidence):
- H_A finds parameters in high-activity components (threshold-based)
- C_A measures the fraction of (signal, probe) pairs with significant change (continuous)
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from dcaf.core.defaults import TAU_ACT, TAU_COMP


def compute_component_magnitude(
    component: str,
    magnitudes_by_signal: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute aggregated activation magnitude for a component.

    m_agg^(k) = (1/|T|) Σ_i m_i^(k)

    Args:
        component: Component identifier
        magnitudes_by_signal: {signal_name: {component: magnitude}}

    Returns:
        Aggregated magnitude across all signals
    """
    total = 0.0
    count = 0

    for signal_name, comp_magnitudes in magnitudes_by_signal.items():
        if component in comp_magnitudes:
            total += comp_magnitudes[component]
            count += 1

    return total / count if count > 0 else 0.0


def compute_param_weight(
    param_magnitude: float,
    max_magnitude_in_component: float,
    eps: float = 1e-8,
) -> float:
    """
    Compute parameter weight within its component.

    w_param(p) = |p| / max_{p': μ(p')=μ(p)} |p'|

    Args:
        param_magnitude: Magnitude of this parameter
        max_magnitude_in_component: Max magnitude in same component
        eps: Numerical stability

    Returns:
        Weight ∈ [0, 1]
    """
    if max_magnitude_in_component < eps:
        return 0.0
    return abs(param_magnitude) / max_magnitude_in_component


def compute_activation_discovery_set(
    all_params: Set[int],
    param_to_component: Dict[int, str],
    param_magnitudes: Dict[int, float],
    magnitudes_by_signal: Dict[str, Dict[str, float]],
    tau_comp: float = TAU_COMP,
    tau_act: float = TAU_ACT,
) -> Tuple[Set[int], Dict[int, float]]:
    """
    Compute activation discovery set H_A.

    Two-stage process:
    1. Screen components by aggregated magnitude (70th percentile - generous)
    2. Filter parameters within flagged components (85th percentile - strict)

    Args:
        all_params: Set of all parameter indices
        param_to_component: {param_index: component_id}
        param_magnitudes: {param_index: weight magnitude}
        magnitudes_by_signal: {signal_name: {component: activation_magnitude}}
        tau_comp: Component screening percentile (default 70)
        tau_act: Parameter filtering percentile (default 85)

    Returns:
        (H_A set of param indices, {param_index: S_A score})
    """
    # Get all components
    components = set(param_to_component.values())

    # Stage 1: Compute m_agg for each component
    m_agg: Dict[str, float] = {}
    for component in components:
        m_agg[component] = compute_component_magnitude(
            component, magnitudes_by_signal
        )

    if not m_agg:
        return set(), {}

    # Component screening threshold (70th percentile - generous)
    m_values = list(m_agg.values())
    max_m = max(m_values)
    if max_m <= 0:
        return set(), {}

    threshold_comp = np.percentile(m_values, tau_comp)
    if threshold_comp <= 0:
        threshold_comp = max_m * 0.01

    K_flagged = {k for k, m in m_agg.items() if m >= threshold_comp}

    if not K_flagged:
        return set(), {}

    # Stage 2: Compute S_A for parameters in flagged components
    # First, compute max magnitude per component for w_param
    max_mag_per_component: Dict[str, float] = {}
    for p, component in param_to_component.items():
        mag = param_magnitudes.get(p, 0.0)
        if component not in max_mag_per_component:
            max_mag_per_component[component] = mag
        else:
            max_mag_per_component[component] = max(
                max_mag_per_component[component], mag
            )

    S_A: Dict[int, float] = {}
    flagged_scores: List[float] = []

    for p in all_params:
        component = param_to_component.get(p)
        if component is None:
            S_A[p] = 0.0
            continue

        if component in K_flagged:
            # S_A = (m_agg / max_m) * w_param
            m_ratio = m_agg[component] / max_m if max_m > 0 else 0
            w_param = compute_param_weight(
                param_magnitudes.get(p, 0.0),
                max_mag_per_component.get(component, 1.0),
            )
            score = m_ratio * w_param
            S_A[p] = score
            flagged_scores.append(score)
        else:
            S_A[p] = 0.0

    if not flagged_scores or max(flagged_scores) <= 0:
        return set(), S_A

    # Parameter threshold within flagged (85th percentile)
    threshold_param = np.percentile(flagged_scores, tau_act)
    if threshold_param <= 0:
        threshold_param = max(flagged_scores) * 0.01

    H_A = {
        p for p, score in S_A.items()
        if param_to_component.get(p) in K_flagged and score >= threshold_param
    }

    return H_A, S_A


def compute_activation_discovery_scores(
    all_params: Set[int],
    param_to_component: Dict[int, str],
    param_magnitudes: Dict[int, float],
    magnitudes_by_signal: Dict[str, Dict[str, float]],
) -> Dict[int, float]:
    """
    Compute S_A scores for all parameters.

    Convenience wrapper that returns just the scores.

    Args:
        all_params: Set of all parameter indices
        param_to_component: {param_index: component_id}
        param_magnitudes: {param_index: weight magnitude}
        magnitudes_by_signal: {signal_name: {component: activation_magnitude}}

    Returns:
        {param_index: S_A score}
    """
    _, S_A = compute_activation_discovery_set(
        all_params=all_params,
        param_to_component=param_to_component,
        param_magnitudes=param_magnitudes,
        magnitudes_by_signal=magnitudes_by_signal,
    )
    return S_A


__all__ = [
    "compute_component_magnitude",
    "compute_param_weight",
    "compute_activation_discovery_set",
    "compute_activation_discovery_scores",
]
