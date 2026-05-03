"""
Projection-level weight confidence, aggregated to component level.

def:projection-level-weight-confidence; def:component-level-weight-confidence:
  C_W^(proj) = min(1, presence + α·opp_deg) × baseline_filter
  C_W^(k) = max_{proj ∈ component} C_W^(proj)
"""

from typing import Dict, List

from dcaf.core.defaults import ALPHA, TAU_BASE, TAU_SIG, Q

from .significance import sig, sig_bar


def compute_projection_confidence(
    proj: str,
    rms_by_signal: Dict[str, Dict[str, float]],
    effectiveness: Dict[str, float],
    opp_degree: float,
    behavioral_signals: List[str],
    baseline_signals: List[str],
    alpha: float = ALPHA,
    q: int = Q,
    tau_sig: float = TAU_SIG,
    tau_base: float = TAU_BASE,
) -> float:
    """
    Compute weight confidence for a single projection.

    C_W^(proj) = min(1, presence + α·opp_deg) × baseline_filter

    Args:
        proj: Projection ID
        rms_by_signal: {signal_id: {proj_id: rms_norm}} — RMS norms per signal
        effectiveness: {signal_id: eff_value}
        opp_degree: Opposition degree for this projection
        behavioral_signals: Signal IDs in T+ ∪ T-
        baseline_signals: Signal IDs in T⁰
        alpha: Opposition bonus weight
        q: Effectiveness power exponent
        tau_sig: Significance percentile threshold
        tau_base: Baseline percentile threshold

    Returns:
        C_W value in [0, 1]
    """
    # Signal presence: effectiveness-weighted fraction of significant signals
    numerator = 0.0
    denominator = 0.0

    for sig_id in behavioral_signals:
        eff = effectiveness.get(sig_id, 1.0)
        eff_q = eff ** q

        if sig_id in rms_by_signal:
            is_sig = sig(proj, rms_by_signal[sig_id], tau_sig)
            if is_sig:
                numerator += eff_q

        denominator += eff_q

    presence = numerator / denominator if denominator > 0 else 0.0

    # Opposition bonus
    score = min(1.0, presence + alpha * opp_degree)

    # Baseline filter: must be insignificant under ALL neutral signals
    for sig_id in baseline_signals:
        if sig_id in rms_by_signal:
            if not sig_bar(proj, rms_by_signal[sig_id], tau_base):
                return 0.0

    return score


def aggregate_component_confidence(
    component_projs: List[str],
    proj_confidences: Dict[str, float],
) -> float:
    """
    Aggregate projection confidences to component level via max.

    C_W^(k) = max_{proj ∈ component} C_W^(proj)

    Args:
        component_projs: List of projection IDs for this component
        proj_confidences: {proj_id: C_W_proj}

    Returns:
        Component-level C_W
    """
    if not component_projs:
        return 0.0
    return max(proj_confidences.get(p, 0.0) for p in component_projs)


__all__ = [
    "compute_projection_confidence",
    "aggregate_component_confidence",
]
