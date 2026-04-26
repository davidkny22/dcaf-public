"""
Cross-validation utilities for weight and activation criteria (§5, §4).

Identifies parameters where BOTH weight deltas AND activation deltas
pass their respective criteria, providing cross-domain signal aggregation
per §3.2 (Multi-Path Discovery).
"""

from __future__ import annotations

import re
from typing import Dict, Set, List, Callable, Any, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from dcaf.domains.weight.criteria import ParamCriteriaEngine
    from dcaf.domains.activation.criteria import ActivationCriteriaEngine

from dcaf.confidence.signals import aggregate_cross_validated_signals


def cross_validate_criteria(
    weight_engine: ParamCriteriaEngine,
    activation_engine: ActivationCriteriaEngine,
    weight_criteria: str,
    activation_criteria: str,
    component_mapper: Callable[[str], str],
) -> Dict[str, Any]:
    """
    Cross-validate weight and activation criteria with signal aggregation.

    Finds parameters where BOTH:
    - Weight delta passes weight_criteria
    - Corresponding component's activation delta passes activation_criteria

    Args:
        weight_engine: ParamCriteriaEngine for weight deltas
        activation_engine: ActivationCriteriaEngine for activation deltas
        weight_criteria: Criteria expression for weights
        activation_criteria: Criteria expression for activations
        component_mapper: Function mapping param_name -> component_id

    Returns:
        Dict with cross-validation statistics, matching items, and combined signals:
        {
            "weight_only": int,
            "activation_only": int,
            "cross_validated": int,
            "cross_validated_items": [
                {
                    "parameter": str,
                    "component": str,
                    "weight_signals": List[str],
                    "activation_signals": List[str],
                    "combined_signal_count": int,
                    "combined_confidence": float,  # 0.0-1.0
                    "common_signals": List[str],  # Signals in BOTH domains
                }
            ],
            "avg_combined_confidence": float,  # Average across all cross-validated
        }
    """
    # Find matching weights and components with signal details
    matching_params, weight_signal_details = weight_engine.find_matching_params(weight_criteria)
    matching_components, activation_signal_details = activation_engine.find_matching_components(activation_criteria)

    # Get total available signals for each domain
    total_available_weight = len(weight_engine.deltas)
    total_available_activation = len(activation_engine.activation_deltas)

    # Map parameters to components
    param_to_component = {}
    for param in matching_params:
        component = component_mapper(param)
        param_to_component[param] = component

    # Find cross-validated matches and aggregate signals
    cross_validated = []
    total_combined_confidence = 0.0

    for param, component in param_to_component.items():
        if component in matching_components:
            # Get signal info from both domains
            weight_sig_info = weight_signal_details[param]
            activation_sig_info = activation_signal_details[component]

            # Aggregate signals using confidence utility
            combined = aggregate_cross_validated_signals(
                weight_signals=weight_sig_info["signals"],
                activation_signals=activation_sig_info["signals"],
                total_available_weight=total_available_weight,
                total_available_activation=total_available_activation,
            )

            cross_validated.append({
                "parameter": param,
                "component": component,
                "weight_signals": sorted(list(weight_sig_info["signals"])),
                "activation_signals": sorted(list(activation_sig_info["signals"])),
                **combined,  # Unpack: weight_signal_count, activation_signal_count,
                             # combined_signal_count, combined_confidence, common_signals
            })

            total_combined_confidence += combined["combined_confidence"]

    # Compute average combined confidence
    avg_combined_confidence = (
        total_combined_confidence / len(cross_validated) if cross_validated else 0.0
    )

    return {
        "weight_only": len(matching_params) - len(cross_validated),
        "activation_only": len(matching_components) - len(cross_validated),
        "cross_validated": len(cross_validated),
        "cross_validated_items": cross_validated,
        "avg_combined_confidence": avg_combined_confidence,
    }


def map_component_to_parameters(
    component_id: str,
    all_param_names: List[str],
) -> List[str]:
    """
    Map component ID to corresponding parameter names.

    Args:
        component_id: e.g., "L10H3", "L10_MLP"
        all_param_names: All parameter names in model

    Returns:
        List of parameters belonging to this component
    """
    layer_match = re.search(r"L(\d+)", component_id)
    if not layer_match:
        return []

    layer_num = layer_match.group(1)

    # Build patterns based on component type
    if "_MLP" in component_id:
        patterns = [
            f"layers.{layer_num}.mlp",
            f"layers.{layer_num}.feed_forward",
            f"h.{layer_num}.mlp",
        ]
    elif "H" in component_id:
        patterns = [
            f"layers.{layer_num}.self_attn",
            f"layers.{layer_num}.attention",
            f"h.{layer_num}.attn",
        ]
    else:
        patterns = [f"layers.{layer_num}"]

    matching = []
    for param_name in all_param_names:
        if any(pattern in param_name for pattern in patterns):
            matching.append(param_name)

    return matching


def param_to_component(param_name: str) -> str:
    """
    Map parameter name to component ID.

    This is a convenience function for common architectures.
    For custom mappings, users can define their own mapper function.

    Examples:
        "model.layers.10.mlp.down_proj.weight" -> "L10_MLP"
        "model.layers.10.self_attn.q_proj.weight" -> "L10_ATTN"
        "h.5.mlp.c_fc.weight" -> "L5_MLP"

    Args:
        param_name: Full parameter name

    Returns:
        Component ID (e.g., "L10_MLP", "L10_ATTN")
    """
    # Extract layer number
    layer_match = re.search(r'(?:layers?|h)\.(\d+)', param_name)
    if not layer_match:
        return "UNKNOWN"

    layer_num = int(layer_match.group(1))

    # Determine component type
    if 'mlp' in param_name.lower() or 'feed_forward' in param_name.lower():
        return f"L{layer_num}_MLP"
    elif 'attn' in param_name.lower() or 'attention' in param_name.lower():
        # For attention, we use L{N}_ATTN for entire attention block
        # Could be more granular (Q/K/V/O) if needed
        return f"L{layer_num}_ATTN"
    else:
        # Catch-all for layer norm, etc.
        return f"L{layer_num}"


__all__ = [
    "cross_validate_criteria",
    "map_component_to_parameters",
    "param_to_component",
]
