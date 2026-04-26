"""Circuit graph reconstruction and identification (§9, Def 9.9).

Public API:
- CircuitGraph / CircuitNode / CircuitEdge: Graph data structures
- PathwayAttribution: Q/K/V pathway attribution for attention edges
- FunctionalCategory / TieredClassification / classify_component_tiered: Component classification (Def 11.23-11.27)
- Circuit / CircuitValidation / CircuitAnalysisResults: Result types
- CircuitIdentifier: 7-step circuit identification pipeline
- KnownCircuitsDatabase / KnownCircuit / CircuitType: Pre-identified circuit database
- SteeringVector / SteeringAnalysis: Steering vector optimization (§10)
- ComponentResult: Unified per-component result container (§13, Def 13.4)
"""

from dcaf.circuit.graph import CircuitGraph, CircuitNode, CircuitEdge
from dcaf.circuit.pathway import PathwayAttribution, compute_pathway_attribution, compute_pathway_from_weight_deltas
from dcaf.circuit.classification import (
    FunctionalCategory,
    ComponentClassification,
    TieredClassification,
    classify_component,
    classify_component_tiered,
    classify_component_detailed,
    classify_from_impact,
    classify_circuit,
    classify_all_components,
    get_recognition_components,
    get_steering_components,
    get_preference_components,
    get_shared_components,
    get_false_positive_components,
    get_classification_summary,
)
from dcaf.circuit.results import Circuit, CircuitValidation, CircuitAnalysisResults
from dcaf.circuit.identifier import CircuitIdentifier
from dcaf.circuit.known_circuits import KnownCircuitsDatabase, KnownCircuit, CircuitType
from dcaf.circuit.steering import (
    SteeringVector,
    SteeringAlignment,
    SteeringAnalysis,
    compute_full_steering_analysis,
)
from dcaf.circuit.component_result import ComponentResult

__all__ = [
    # Graph structures
    "CircuitGraph",
    "CircuitNode",
    "CircuitEdge",
    # Pathway attribution
    "PathwayAttribution",
    "compute_pathway_attribution",
    "compute_pathway_from_weight_deltas",
    # Classification (Def 11.23-11.27)
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
    # Results
    "Circuit",
    "CircuitValidation",
    "CircuitAnalysisResults",
    # Identifier
    "CircuitIdentifier",
    # Known circuits database
    "KnownCircuitsDatabase",
    "KnownCircuit",
    "CircuitType",
    # Steering (§10)
    "SteeringVector",
    "SteeringAlignment",
    "SteeringAnalysis",
    "compute_full_steering_analysis",
    # Component result (§13)
    "ComponentResult",
]
