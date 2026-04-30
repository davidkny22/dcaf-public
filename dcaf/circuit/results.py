"""
Result dataclasses for circuit analysis (§9, Def 9.9).

Provides structured types for circuits, validation, and analysis results.
For activation-related types (ActivationSnapshot, ActivationDelta),
see dcaf.domains.activation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

from dcaf.core.defaults import ATTENTION_WEIGHT, TAU_EDGE

if TYPE_CHECKING:
    from dcaf.circuit.graph import CircuitEdge, CircuitNode
    from dcaf.ablation.results import WeightClassification


__all__ = [
    "CircuitValidation",
    "Circuit",
    "CircuitAnalysisResults",
]


@dataclass
class CircuitValidation:
    """
    Validation results for a circuit.

    Tests whether the circuit exhibits superadditive effects - where ablating
    all weights together has a larger impact than the sum of individual ablations.
    """

    individual_ablation_impact: float  # Sum of individual weight ablation effects
    whole_circuit_ablation_impact: float  # Effect of ablating all weights together
    superadditive: bool  # whole > sum of individuals * threshold?

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "individual_ablation_impact": self.individual_ablation_impact,
            "whole_circuit_ablation_impact": self.whole_circuit_ablation_impact,
            "superadditive": self.superadditive,
        }


@dataclass
class Circuit:
    """
    A connected computational pathway for safety behavior.

    Represents a circuit of components (attention heads, MLP layers) connected
    by weight parameters, with information flow direction and validation.
    """

    name: str

    # Components in the circuit
    components: List[str] = field(default_factory=list)  # ["L10H3", "L10_MLP", "L12H7"]

    # Weights connecting the components
    weight_params: List[str] = field(default_factory=list)  # Full parameter names

    # Information flow (topologically sorted)
    flow: List[str] = field(default_factory=list)  # ["L10H3", "L10_MLP", "L12H7"]

    # Graph structure
    edges: List["CircuitEdge"] = field(default_factory=list)
    nodes: Dict[str, "CircuitNode"] = field(default_factory=dict)  # Node properties

    # Metadata
    clustering_method: str = "disjoint"  # "disjoint", "probe-response", "functional"

    # Validation
    validation: Optional[CircuitValidation] = None

    # Confidence score
    confidence: float = 0.0

    # Weight classifications (multi-probe)
    weight_classifications: Dict[str, "WeightClassification"] = field(default_factory=dict)
    circuit_type: Optional[str] = None  # "recognition", "generation", "shared", "mixed"

    def compute_circuit_type(self) -> str:
        """
        Compute circuit type from majority (>60%) of weight classifications.

        Returns:
            "recognition" if >60% weights are recognition-specific
            "generation" if >60% weights are generation-specific
            "shared" if >60% weights are shared
            "mixed" if no clear majority
        """
        if not self.weight_classifications:
            return "mixed"

        # Count classification types
        counts = {
            "recognition-specific": 0,
            "generation-specific": 0,
            "shared": 0,
            "false-positive": 0,
        }

        for wc in self.weight_classifications.values():
            classification = wc.classification
            if classification in counts:
                counts[classification] += 1

        total = len(self.weight_classifications)
        if total == 0:
            return "mixed"

        # Check for majority (>60%)
        threshold = 0.6
        for cls_type in ["recognition-specific", "generation-specific", "shared"]:
            if counts[cls_type] / total > threshold:
                # Map to simple circuit type names
                if cls_type == "recognition-specific":
                    return "recognition"
                elif cls_type == "generation-specific":
                    return "generation"
                else:
                    return "shared"

        return "mixed"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "components": self.components,
            "weight_params": self.weight_params,
            "flow": self.flow,
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "edge_type": e.edge_type,
                }
                for e in self.edges
            ],
            "clustering_method": self.clustering_method,
            "validation": self.validation.to_dict() if self.validation else None,
            "confidence": self.confidence,
            "weight_classifications": {
                name: wc.to_dict() for name, wc in self.weight_classifications.items()
            },
            "circuit_type": self.circuit_type,
        }


@dataclass
class CircuitAnalysisResults:
    """
    Aggregated results from circuit analysis.

    Contains all identified circuits, metadata about the analysis, and
    references to the activation snapshots used.
    """

    circuits: List[Circuit] = field(default_factory=list)
    clustering_method: str = "disjoint"
    edge_threshold: float = TAU_EDGE
    attention_weight: float = ATTENTION_WEIGHT
    total_weight_candidates: int = 0
    probe_set_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        import json

        data = self.to_dict()
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CircuitAnalysisResults":
        """Load results from JSON file."""
        import json

        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "circuits": [c.to_dict() for c in self.circuits],
            "metadata": {
                "clustering_method": self.clustering_method,
                "edge_threshold": self.edge_threshold,
                "attention_weight": self.attention_weight,
                "total_weight_candidates": self.total_weight_candidates,
                "circuits_identified": len(self.circuits),
                "probe_set_name": self.probe_set_name,
                "timestamp": self.timestamp,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitAnalysisResults":
        """Load from dictionary."""
        from dcaf.circuit.graph import CircuitEdge

        circuits = []
        for c_data in data.get("circuits", []):
            edges = [
                CircuitEdge(
                    source=e["source"],
                    target=e["target"],
                    weight=e["weight"],
                    edge_type=e["edge_type"],
                )
                for e in c_data.get("edges", [])
            ]
            validation = None
            if c_data.get("validation"):
                v = c_data["validation"]
                validation = CircuitValidation(
                    individual_ablation_impact=v["individual_ablation_impact"],
                    whole_circuit_ablation_impact=v["whole_circuit_ablation_impact"],
                    superadditive=v["superadditive"],
                )
            # Restore weight classifications if present
            weight_classifications = {}
            if c_data.get("weight_classifications"):
                from dcaf.ablation.results import WeightClassification
                for name, wc_data in c_data["weight_classifications"].items():
                    weight_classifications[name] = WeightClassification.from_dict(wc_data)

            circuits.append(
                Circuit(
                    name=c_data["name"],
                    components=c_data.get("components", []),
                    weight_params=c_data.get("weight_params", []),
                    flow=c_data.get("flow", []),
                    edges=edges,
                    clustering_method=c_data.get("clustering_method", "disjoint"),
                    validation=validation,
                    confidence=c_data.get("confidence", 0.0),
                    weight_classifications=weight_classifications,
                    circuit_type=c_data.get("circuit_type"),
                )
            )

        metadata = data.get("metadata", {})
        return cls(
            circuits=circuits,
            clustering_method=metadata.get("clustering_method", "disjoint"),
            edge_threshold=metadata.get("edge_threshold", TAU_EDGE),
            attention_weight=metadata.get("attention_weight", ATTENTION_WEIGHT),
            total_weight_candidates=metadata.get("total_weight_candidates", 0),
            probe_set_name=metadata.get("probe_set_name", ""),
            timestamp=metadata.get("timestamp", ""),
        )
