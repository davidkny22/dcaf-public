"""Weight-based discovery path H_W.

Spec: def:weight-based-discovery-score; def:weight-based-discovery-set.

Identifies projections that changed significantly during behavioral training.
S_W uses the same formula as C_W — discovery and confidence are the same
computation for the weight domain. The difference is conceptual:
  S_W determines what to analyze (discovery threshold)
  C_W measures how confident we are (continuous score)
"""

from typing import Dict, List, Set, Tuple

from dcaf.core.defaults import TAU_W_DEFAULT
from dcaf.domains.weight.confidence import (
    compute_projection_confidence,
)


def compute_weight_discovery_scores(
    rms_by_signal: Dict[str, Dict[str, float]],
    effectiveness: Dict[str, float],
    opp_degrees: Dict[str, float],
    behavioral_signals: List[str],
    baseline_signals: List[str],
    alpha: float = 0.2,
    q: int = 2,
    tau_sig: float = 85.0,
    tau_base: float = 50.0,
) -> Dict[str, float]:
    """Compute S_W for all projections (def:weight-based-discovery-score).

    Args:
        rms_by_signal: {signal_id: {proj_id: rms_norm}} — RMS norms per projection per signal
        effectiveness: {signal_id: eff_value}
        opp_degrees: {proj_id: opposition_degree}
        behavioral_signals: Signal IDs in T+ union T-
        baseline_signals: Signal IDs in T0

    Returns:
        {proj_id: S_W score}
    """
    all_projs = set()
    for sig_rms in rms_by_signal.values():
        all_projs.update(sig_rms.keys())

    scores = {}
    for proj in all_projs:
        scores[proj] = compute_projection_confidence(
            proj=proj,
            rms_by_signal=rms_by_signal,
            effectiveness=effectiveness,
            opp_degree=opp_degrees.get(proj, 0.0),
            behavioral_signals=behavioral_signals,
            baseline_signals=baseline_signals,
            alpha=alpha,
            q=q,
            tau_sig=tau_sig,
            tau_base=tau_base,
        )
    return scores


def compute_weight_discovery_set(
    rms_by_signal: Dict[str, Dict[str, float]],
    effectiveness: Dict[str, float],
    opp_degrees: Dict[str, float],
    behavioral_signals: List[str],
    baseline_signals: List[str],
    tau_W: float = TAU_W_DEFAULT,
    **kwargs,
) -> Tuple[Set[str], Dict[str, float]]:
    """Compute H_W = {proj : S_W(proj) >= tau_W} (def:weight-based-discovery-set).

    Returns:
        (H_W set of projection IDs, {proj_id: S_W score})
    """
    S_W = compute_weight_discovery_scores(
        rms_by_signal=rms_by_signal,
        effectiveness=effectiveness,
        opp_degrees=opp_degrees,
        behavioral_signals=behavioral_signals,
        baseline_signals=baseline_signals,
        **kwargs,
    )
    H_W = {proj for proj, score in S_W.items() if score >= tau_W}
    return H_W, S_W


__all__ = [
    "compute_weight_discovery_scores",
    "compute_weight_discovery_set",
]
