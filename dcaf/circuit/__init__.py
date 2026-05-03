"""Circuit graph reconstruction and identification (sec:circuit-graph).

Torch-backed pathway, identifier, and steering helpers are loaded lazily so
``import dcaf.circuit`` remains a lightweight dataclass/graph import.
"""

from dcaf.circuit.classification import (
    ComponentClassification,
    FunctionalCategory,
    TieredClassification,
    classify_all_components,
    classify_circuit,
    classify_component,
    classify_component_detailed,
    classify_component_tiered,
    classify_from_impact,
    get_classification_summary,
    get_false_positive_components,
    get_preference_components,
    get_recognition_components,
    get_shared_components,
    get_steering_components,
)
from dcaf.circuit.component_result import ComponentResult
from dcaf.circuit.graph import CircuitEdge, CircuitGraph, CircuitNode
from dcaf.circuit.known_circuits import CircuitType, KnownCircuit, KnownCircuitsDatabase
from dcaf.circuit.results import Circuit, CircuitAnalysisResults, CircuitValidation

_LAZY_EXPORTS = {
    "PathwayAttribution": ("dcaf.circuit.pathway", "PathwayAttribution"),
    "compute_pathway_attribution": (
        "dcaf.circuit.pathway",
        "compute_pathway_attribution",
    ),
    "compute_pathway_from_weight_deltas": (
        "dcaf.circuit.pathway",
        "compute_pathway_from_weight_deltas",
    ),
    "CircuitIdentifier": ("dcaf.circuit.identifier", "CircuitIdentifier"),
    "SteeringVector": ("dcaf.circuit.steering", "SteeringVector"),
    "SteeringAlignment": ("dcaf.circuit.steering", "SteeringAlignment"),
    "SteeringAnalysis": ("dcaf.circuit.steering", "SteeringAnalysis"),
    "compute_full_steering_analysis": (
        "dcaf.circuit.steering",
        "compute_full_steering_analysis",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "CircuitGraph",
    "CircuitNode",
    "CircuitEdge",
    "FunctionalCategory",
    "ComponentClassification",
    "TieredClassification",
    "classify_component",
    "classify_component_tiered",
    "classify_component_detailed",
    "classify_from_impact",
    "classify_circuit",
    "classify_all_components",
    "get_recognition_components",
    "get_steering_components",
    "get_preference_components",
    "get_shared_components",
    "get_false_positive_components",
    "get_classification_summary",
    "Circuit",
    "CircuitValidation",
    "CircuitAnalysisResults",
    "KnownCircuitsDatabase",
    "KnownCircuit",
    "CircuitType",
    "ComponentResult",
    *_LAZY_EXPORTS.keys(),
]
