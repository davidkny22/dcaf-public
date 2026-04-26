"""In-memory per-component output assembler (§13, Def 13.4).

Assembles per-component enhanced output with:
1. Unified confidence C^(k) with domain breakdown
2. Domain dominance, deviation, and disagreement scores
3. Discovery path attribution
4. Functional classification (primary/auxiliary tiers)
5. Interaction metadata (SOLO/PAIR/GATE + type)
6. Bidirectionality status
7. Projection-level C_W breakdown

For the full-run JSON schema assembler, see dcaf.output.schema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentOutput:
    """Enhanced per-component output (Def 13.4)."""

    component: str
    confidence: float = 0.0
    domain_scores: Dict[str, float] = field(default_factory=dict)
    domain_dominance: Dict[str, float] = field(default_factory=dict)
    domain_deviation: Dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0
    discovery_paths: List[str] = field(default_factory=list)
    path_count: int = 0
    multi_path_bonus: float = 0.0
    classification: Optional[Dict[str, Any]] = None
    interaction: Optional[Dict[str, Any]] = None
    bidirectional: bool = False
    projection_breakdown: Optional[Dict[str, Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "confidence": self.confidence,
            "domain_scores": self.domain_scores,
            "domain_dominance": self.domain_dominance,
            "domain_deviation": self.domain_deviation,
            "disagreement": self.disagreement,
            "discovery": {
                "paths": self.discovery_paths,
                "path_count": self.path_count,
                "bonus": self.multi_path_bonus,
            },
            "classification": self.classification,
            "interaction": self.interaction,
            "bidirectional": self.bidirectional,
            "projection_breakdown": self.projection_breakdown,
        }


def assemble_component_output(
    component: str,
    C_W: float = 0.0,
    C_A: float = 0.0,
    C_G: float = 0.0,
    unified_confidence: float = 0.0,
    paths: Optional[List[str]] = None,
    bonus: float = 0.0,
    classification: Optional[Dict[str, Any]] = None,
    interaction: Optional[Dict[str, Any]] = None,
    bidirectional: bool = False,
    projection_breakdown: Optional[Dict[str, Dict[str, float]]] = None,
) -> ComponentOutput:
    """Assemble a single component's output."""
    C_total = C_W + C_A + C_G
    C_bar = C_total / 3.0 if C_total > 0 else 0.0

    if C_total > 0:
        dominance = {"weight": C_W / C_total, "activation": C_A / C_total, "geometry": C_G / C_total}
    else:
        dominance = {"weight": 0.33, "activation": 0.33, "geometry": 0.34}

    deviation = {
        "weight": C_W - C_bar,
        "activation": C_A - C_bar,
        "geometry": C_G - C_bar,
    }
    disagreement = sum(v**2 for v in deviation.values()) / 3.0

    return ComponentOutput(
        component=component,
        confidence=unified_confidence,
        domain_scores={"C_W": C_W, "C_A": C_A, "C_G": C_G},
        domain_dominance=dominance,
        domain_deviation=deviation,
        disagreement=disagreement,
        discovery_paths=paths or [],
        path_count=len(paths) if paths else 0,
        multi_path_bonus=bonus,
        classification=classification,
        interaction=interaction,
        bidirectional=bidirectional,
        projection_breakdown=projection_breakdown,
    )


def assemble_output(
    components: Dict[str, ComponentOutput],
    circuit_graph: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble complete DCAF output."""
    return {
        "version": "0.1.0",
        "components": {c: out.to_dict() for c, out in components.items()},
        "circuit_graph": circuit_graph,
        "metadata": metadata or {},
        "summary": {
            "total_components": len(components),
            "confirmed": sum(
                1 for c in components.values()
                if c.classification and c.classification.get("status") == "Confirmed"
            ),
            "bidirectional": sum(1 for c in components.values() if c.bidirectional),
        },
    }


__all__ = [
    "ComponentOutput",
    "assemble_component_output",
    "assemble_output",
]
