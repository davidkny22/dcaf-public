"""
Projection-level significance predicates.

Def 2.3 (§2):
  sig(proj, i) = 1[||ΔW_i^(proj)||_RMS >= Φ_{τ_sig}]
  sig_bar(proj, i) = 1[||ΔW_i^(proj)||_RMS < Φ_{τ_base}]

Percentile thresholds computed across all ~700 RMS-normalized projections.
"""

from typing import Dict, List

import numpy as np

from dcaf.core.defaults import TAU_SIG, TAU_BASE


def compute_significance(
    rms_values: Dict[str, float],
    tau_sig: float = TAU_SIG,
) -> Dict[str, bool]:
    """
    Determine which projections are significant.

    sig(proj, i) = 1[rms(proj) >= Φ_{τ_sig}(all rms values)]

    Args:
        rms_values: {proj_id: rms_norm} for one signal, across all projections
        tau_sig: Significance percentile threshold (default 85)

    Returns:
        {proj_id: True if significant}
    """
    all_norms = list(rms_values.values())
    if not all_norms:
        return {}
    threshold = float(np.percentile(all_norms, tau_sig))
    return {proj: norm >= threshold for proj, norm in rms_values.items()}


def compute_baseline_insignificance(
    rms_values: Dict[str, float],
    tau_base: float = TAU_BASE,
) -> Dict[str, bool]:
    """
    Determine which projections are below baseline (insignificant).

    sig_bar(proj, i) = 1[rms(proj) < Φ_{τ_base}(all rms values)]

    Args:
        rms_values: {proj_id: rms_norm} for one baseline signal
        tau_base: Baseline percentile threshold (default 50)

    Returns:
        {proj_id: True if below baseline}
    """
    all_norms = list(rms_values.values())
    if not all_norms:
        return {}
    threshold = float(np.percentile(all_norms, tau_base))
    return {proj: norm < threshold for proj, norm in rms_values.items()}


def sig(
    proj: str,
    rms_values: Dict[str, float],
    tau_sig: float = TAU_SIG,
) -> bool:
    """
    Check if a single projection is significant.

    Args:
        proj: Projection ID
        rms_values: {proj_id: rms_norm} across all projections for one signal
        tau_sig: Significance percentile threshold

    Returns:
        True if projection's RMS norm >= tau_sig percentile
    """
    all_norms = list(rms_values.values())
    if not all_norms:
        return False
    threshold = float(np.percentile(all_norms, tau_sig))
    return rms_values.get(proj, 0.0) >= threshold


def sig_bar(
    proj: str,
    rms_values: Dict[str, float],
    tau_base: float = TAU_BASE,
) -> bool:
    """
    Check if a single projection is below baseline (insignificant).

    Args:
        proj: Projection ID
        rms_values: {proj_id: rms_norm} across all projections for one signal
        tau_base: Baseline percentile threshold

    Returns:
        True if projection's RMS norm < tau_base percentile
    """
    all_norms = list(rms_values.values())
    if not all_norms:
        return True
    threshold = float(np.percentile(all_norms, tau_base))
    return rms_values.get(proj, 0.0) < threshold


__all__ = [
    "compute_significance",
    "compute_baseline_insignificance",
    "sig",
    "sig_bar",
]
