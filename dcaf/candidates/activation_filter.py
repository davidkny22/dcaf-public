"""
Activation domain filtering for candidate validation.

Filters candidates by activation confidence (C_A).

Activation filtering validates that candidates show consistent
activation pattern changes across training signals and probe types.
"""

from typing import Dict, Set, List, Any, Optional, Tuple

from dcaf.core.defaults import TAU_A_DEFAULT


def filter_by_activation_confidence(
    candidates: Set[Any],
    activation_confidences: Dict[Any, float],
    component_map: Dict[Any, Any],
    tau_A: float = TAU_A_DEFAULT,
) -> Set[Any]:
    """
    Filter candidates by activation confidence threshold.

    {p ∈ candidates : C_A⁽μ⁽ᵖ⁾⁾ ≥ τ_A}

    Args:
        candidates: Set of candidate param_ids to filter
        activation_confidences: {component_id: C_A} mapping
        component_map: {param_id: component_id} mapping
        tau_A: Activation confidence threshold

    Returns:
        Set of param_ids that pass threshold
    """
    passed = set()

    for param_id in candidates:
        component = component_map.get(param_id)
        if component is None:
            # No component mapping - include by default
            passed.add(param_id)
            continue

        c_a = activation_confidences.get(component)
        if c_a is None or c_a >= tau_A:
            passed.add(param_id)

    return passed


def filter_components_by_activation(
    component_confidences: Dict[Any, float],
    tau_A: float = TAU_A_DEFAULT,
) -> Set[Any]:
    """
    Filter components directly by activation confidence.

    Args:
        component_confidences: {component_id: C_A} mapping
        tau_A: Activation confidence threshold

    Returns:
        Set of component_ids that pass threshold
    """
    return {
        component_id for component_id, c_a in component_confidences.items()
        if c_a >= tau_A
    }


def rank_by_activation_confidence(
    activation_confidences: Dict[Any, float],
) -> List[Tuple[Any, float]]:
    """
    Rank components by activation confidence.

    Args:
        activation_confidences: {component_id: C_A} mapping

    Returns:
        [(component_id, C_A), ...] sorted by C_A descending
    """
    return sorted(
        activation_confidences.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def get_activation_confidence_for_params(
    param_ids: Set[Any],
    activation_confidences: Dict[Any, float],
    component_map: Dict[Any, Any],
) -> Dict[Any, Optional[float]]:
    """
    Get activation confidence for each parameter via component lookup.

    Args:
        param_ids: Set of parameter IDs
        activation_confidences: {component_id: C_A} mapping
        component_map: {param_id: component_id} mapping

    Returns:
        {param_id: C_A or None} mapping
    """
    result = {}
    for param_id in param_ids:
        component = component_map.get(param_id)
        if component is not None:
            result[param_id] = activation_confidences.get(component)
        else:
            result[param_id] = None
    return result


__all__ = [
    "filter_by_activation_confidence",
    "filter_components_by_activation",
    "rank_by_activation_confidence",
    "get_activation_confidence_for_params",
]
