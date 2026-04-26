"""
Geometry domain filtering for candidate validation.

Filters candidates by geometry confidence (C_G).

Geometry filtering validates that candidates exhibit consistent
representational structure (linear representations, direction coherence).
"""

from typing import Dict, Set, List, Any, Optional, Tuple

from dcaf.core.defaults import TAU_G_DEFAULT


def filter_by_geometry_confidence(
    candidates: Set[Any],
    geometry_confidences: Dict[Any, float],
    component_map: Dict[Any, Any],
    tau_G: float = TAU_G_DEFAULT,
) -> Set[Any]:
    """
    Filter candidates by geometry confidence threshold.

    {p ∈ candidates : C_G⁽μ⁽ᵖ⁾⁾ ≥ τ_G}

    Args:
        candidates: Set of candidate param_ids to filter
        geometry_confidences: {component_id: C_G} mapping
        component_map: {param_id: component_id} mapping
        tau_G: Geometry confidence threshold

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

        c_g = geometry_confidences.get(component)
        if c_g is None or c_g >= tau_G:
            passed.add(param_id)

    return passed


def filter_components_by_geometry(
    component_confidences: Dict[Any, float],
    tau_G: float = TAU_G_DEFAULT,
) -> Set[Any]:
    """
    Filter components directly by geometry confidence.

    Args:
        component_confidences: {component_id: C_G} mapping
        tau_G: Geometry confidence threshold

    Returns:
        Set of component_ids that pass threshold
    """
    return {
        component_id for component_id, c_g in component_confidences.items()
        if c_g >= tau_G
    }


def rank_by_geometry_confidence(
    geometry_confidences: Dict[Any, float],
) -> List[Tuple[Any, float]]:
    """
    Rank components by geometry confidence.

    Args:
        geometry_confidences: {component_id: C_G} mapping

    Returns:
        [(component_id, C_G), ...] sorted by C_G descending
    """
    return sorted(
        geometry_confidences.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def get_geometry_confidence_for_params(
    param_ids: Set[Any],
    geometry_confidences: Dict[Any, float],
    component_map: Dict[Any, Any],
) -> Dict[Any, Optional[float]]:
    """
    Get geometry confidence for each parameter via component lookup.

    Args:
        param_ids: Set of parameter IDs
        geometry_confidences: {component_id: C_G} mapping
        component_map: {param_id: component_id} mapping

    Returns:
        {param_id: C_G or None} mapping
    """
    result = {}
    for param_id in param_ids:
        component = component_map.get(param_id)
        if component is not None:
            result[param_id] = geometry_confidences.get(component)
        else:
            result[param_id] = None
    return result


def filter_by_lrs(
    geometry_confidences: Dict[Any, float],
    lrs_scores: Dict[Any, float],
    tau_G: float = TAU_G_DEFAULT,
    tau_lrs: float = 0.4,
) -> Set[Any]:
    """
    Filter by both geometry confidence and LRS score.

    Args:
        geometry_confidences: {component_id: C_G} mapping
        lrs_scores: {component_id: LRS} mapping
        tau_G: Geometry confidence threshold
        tau_lrs: LRS threshold

    Returns:
        Set of component_ids passing both thresholds
    """
    return {
        component_id for component_id, c_g in geometry_confidences.items()
        if c_g >= tau_G and lrs_scores.get(component_id, 0) >= tau_lrs
    }


__all__ = [
    "filter_by_geometry_confidence",
    "filter_components_by_geometry",
    "rank_by_geometry_confidence",
    "get_geometry_confidence_for_params",
    "filter_by_lrs",
]
