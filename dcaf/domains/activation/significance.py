"""
Activation significance predicate.

def:component-significance:
sig_A(k, i, π) = 𝟙[m_i^(k,π) ≥ Φ^A_τact({m_i^(k',π)}_{k'∈K})]
"""

from typing import Dict, List, Set, Tuple

import torch

from dcaf.core.defaults import TAU_ACT


def percentile_threshold_activation(
    magnitudes: Dict[str, float],
    tau_act: float = TAU_ACT,
) -> float:
    """
    Compute τ_act percentile threshold over component magnitudes.

    Φ^A_τact = value at τ_act percentile

    Args:
        magnitudes: {component_id: magnitude} for all components
        tau_act: Percentile threshold (default 85)

    Returns:
        Threshold value
    """
    if not magnitudes:
        return 0.0

    values = torch.tensor(list(magnitudes.values()))
    return torch.quantile(values, tau_act / 100.0).item()


def sig_A(
    component: str,
    magnitude: float,
    threshold: float,
) -> bool:
    """
    Check if component is significant for a (signal, probe) pair.

    sig_A(k, i, π) = 𝟙[m_i^(k,π) ≥ threshold]

    Args:
        component: Component ID
        magnitude: Magnitude value m_i^(k,π)
        threshold: Pre-computed Φ^A_τact threshold

    Returns:
        True if significant
    """
    return magnitude >= threshold


def compute_significance_mask(
    magnitudes: Dict[str, float],
    tau_act: float = TAU_ACT,
) -> Dict[str, bool]:
    """
    Compute significance for all components.

    Args:
        magnitudes: {component_id: magnitude}
        tau_act: Percentile threshold

    Returns:
        {component_id: is_significant}
    """
    threshold = percentile_threshold_activation(magnitudes, tau_act)
    return {
        component: sig_A(component, mag, threshold)
        for component, mag in magnitudes.items()
    }


def get_significant_components(
    magnitudes: Dict[str, float],
    tau_act: float = TAU_ACT,
) -> Set[str]:
    """
    Get set of significant components.

    Args:
        magnitudes: {component_id: magnitude}
        tau_act: Percentile threshold

    Returns:
        Set of component IDs that are significant
    """
    mask = compute_significance_mask(magnitudes, tau_act)
    return {c for c, is_sig in mask.items() if is_sig}


def count_significant(
    magnitudes: Dict[str, float],
    tau_act: float = TAU_ACT,
) -> int:
    """
    Count number of significant components.

    Args:
        magnitudes: {component_id: magnitude}
        tau_act: Percentile threshold

    Returns:
        Count of significant components
    """
    return len(get_significant_components(magnitudes, tau_act))


def rank_by_magnitude(
    magnitudes: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Rank components by magnitude descending.

    Args:
        magnitudes: {component_id: magnitude}

    Returns:
        [(component_id, magnitude), ...] sorted descending
    """
    return sorted(
        magnitudes.items(),
        key=lambda x: x[1],
        reverse=True,
    )


__all__ = [
    "percentile_threshold_activation",
    "sig_A",
    "compute_significance_mask",
    "get_significant_components",
    "count_significant",
    "rank_by_magnitude",
]
