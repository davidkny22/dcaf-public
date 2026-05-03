"""
Unified per-component result dataclass (app:output; def:enhanced-component-output).

ComponentResult combines discovery info, domain scores (C_W, C_A, C_G),
triangulated confidence, functional classification, interaction data,
ablation confirmation, and training/geometry diagnostics into one
output-ready object.

Used as the canonical result container across the full pipeline.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from dcaf.diagnostics.alignment import ActivationDeltaAlignment
    from dcaf.diagnostics.curvature import CurvatureMetrics
    from dcaf.domains.geometry.nonlinear import NonlinearDiagnostics
    from dcaf.domains.geometry.probing import ProbingResults


@dataclass
class ComponentResult:
    """
    Unified result for a single circuit component.

    Assembles all domain scores, classification, interaction data,
    ablation confirmation, and training diagnostics into one output-ready object.

    Attributes:
        component: Component identifier (e.g., "L10_MLP", "L5H3")
        param_names: Parameter names belonging to this component

        # Domain confidence scores
        C_W: Weight domain confidence (or None if not computed)
        C_A: Activation domain confidence (or None)
        C_G: Geometry domain confidence (or None)
        C_unified: Triangulated + multi-path bonus confidence

        # Discovery
        discovery_paths: Set of paths that discovered this ('W', 'A', 'G')
        path_count: Number of discovery paths (1, 2, or 3)
        multi_path_bonus: bonus = beta_path * max(0, path_count - 1)

        # Weight domain details
        bidirectional: Whether component shows bidirectional control
        opp_degree: Opposition degree value

        # Geometry domain details
        lrs: Linear Representation Score
        lrs_breakdown: LRS component values (coh_plus, coh_minus, etc.)

        # Classification (from tiered or simple classification)
        classification: Classification result dict

        # Interaction requirement (how many params needed for effect)
        interaction_requirement: "solo", "pair", or "gate"
        interaction_partners: List of partner component IDs (for pair/gate)

        # Interaction type (how params combine)
        interaction_type: "synergistic", "redundant", "additive", "subadditive"

        # Ablation confirmation
        ablation_confirmed: Whether ablation confirms behavioral relevance
        ablation_status: "behavioral", "general", "none", "untested"

        # Domain diagnostics
        diagnostics: {contributions, deviations, disagreement}

        # Training diagnostics (always computed when available)
        delta_alignment: Activation delta alignment across signal clusters
        curvature: Training path curvature metrics (optional)

        # Nonlinear geometry diagnostics (computed when LRS < threshold)
        nonlinear: PaCMAP silhouettes, Procrustes alignment
        probing: Polynomial and kernel LDA probing results
    """
    component: str
    param_names: List[str] = field(default_factory=list)

    # Domain scores
    C_W: Optional[float] = None
    C_A: Optional[float] = None
    C_G: Optional[float] = None
    C_unified: Optional[float] = None

    # Discovery
    discovery_paths: Set[str] = field(default_factory=set)
    path_count: int = 0
    multi_path_bonus: float = 0.0

    # Weight domain details
    bidirectional: bool = False
    opp_degree: float = 0.0

    # Geometry domain details
    lrs: Optional[float] = None
    lrs_breakdown: Optional[Dict[str, float]] = None

    # Classification
    classification: Optional[Dict[str, Any]] = None

    # Interaction requirement (SOLO/PAIR/GATE)
    interaction_requirement: Optional[str] = None
    interaction_partners: List[str] = field(default_factory=list)

    # Interaction type (SYNERGISTIC/REDUNDANT/etc.)
    interaction_type: Optional[str] = None

    # Ablation
    ablation_confirmed: Optional[bool] = None
    ablation_status: str = "untested"

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Training diagnostics (always computed when available)
    delta_alignment: Optional["ActivationDeltaAlignment"] = None

    # Curvature (optional, computed when config.compute_curvature=True)
    curvature: Optional["CurvatureMetrics"] = None

    # Nonlinear diagnostics (computed when LRS < lrs_nonlinear_threshold)
    nonlinear: Optional["NonlinearDiagnostics"] = None

    # Probing results (computed when LRS < lrs_nonlinear_threshold)
    probing: Optional["ProbingResults"] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        result = {
            "component": self.component,
            "param_names": self.param_names,
            "scores": {
                "C_W": self.C_W,
                "C_A": self.C_A,
                "C_G": self.C_G,
                "C_unified": self.C_unified,
            },
            "discovery": {
                "paths": sorted(self.discovery_paths),
                "path_count": self.path_count,
                "multi_path_bonus": self.multi_path_bonus,
            },
            "weight_details": {
                "bidirectional": self.bidirectional,
                "opp_degree": self.opp_degree,
            },
            "geometry_details": {
                "lrs": self.lrs,
                "lrs_breakdown": self.lrs_breakdown,
            },
            "classification": self.classification,
            "interaction": {
                "requirement": self.interaction_requirement,
                "partners": self.interaction_partners,
                "type": self.interaction_type,
            },
            "ablation": {
                "confirmed": self.ablation_confirmed,
                "status": self.ablation_status,
            },
            "diagnostics": self.diagnostics,
        }

        # Training diagnostics
        if self.delta_alignment is not None:
            result["delta_alignment"] = self.delta_alignment.to_dict()
        else:
            result["delta_alignment"] = None

        if self.curvature is not None:
            result["curvature"] = self.curvature.to_dict()
        else:
            result["curvature"] = None

        # Nonlinear geometry diagnostics
        if self.nonlinear is not None:
            result["nonlinear"] = self.nonlinear.to_dict()
        else:
            result["nonlinear"] = None

        if self.probing is not None:
            result["probing"] = self.probing.to_dict()
        else:
            result["probing"] = None

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentResult":
        """Deserialize from dictionary."""
        scores = data.get("scores", {})
        discovery = data.get("discovery", {})
        weight_details = data.get("weight_details", {})
        geometry_details = data.get("geometry_details", {})
        interaction = data.get("interaction", {})
        ablation = data.get("ablation", {})

        # Reconstruct training diagnostics if present
        delta_alignment = None
        if data.get("delta_alignment") is not None:
            from dcaf.diagnostics.alignment import ActivationDeltaAlignment
            delta_alignment = ActivationDeltaAlignment.from_dict(data["delta_alignment"])

        curvature = None
        if data.get("curvature") is not None:
            from dcaf.diagnostics.curvature import CurvatureMetrics
            curvature = CurvatureMetrics.from_dict(data["curvature"])

        nonlinear = None
        if data.get("nonlinear") is not None:
            from dcaf.domains.geometry.nonlinear import NonlinearDiagnostics
            nonlinear = NonlinearDiagnostics.from_dict(data["nonlinear"])

        probing = None
        if data.get("probing") is not None:
            from dcaf.domains.geometry.probing import ProbingResults
            probing = ProbingResults.from_dict(data["probing"])

        return cls(
            component=data["component"],
            param_names=data.get("param_names", []),
            # Domain scores
            C_W=scores.get("C_W"),
            C_A=scores.get("C_A"),
            C_G=scores.get("C_G"),
            C_unified=scores.get("C_unified"),
            # Discovery
            discovery_paths=set(discovery.get("paths", [])),
            path_count=discovery.get("path_count", 0),
            multi_path_bonus=discovery.get("multi_path_bonus", 0.0),
            # Weight details
            bidirectional=weight_details.get("bidirectional", False),
            opp_degree=weight_details.get("opp_degree", 0.0),
            # Geometry details
            lrs=geometry_details.get("lrs"),
            lrs_breakdown=geometry_details.get("lrs_breakdown"),
            # Classification
            classification=data.get("classification"),
            # Interaction
            interaction_requirement=interaction.get("requirement"),
            interaction_partners=interaction.get("partners", []),
            interaction_type=interaction.get("type"),
            # Ablation
            ablation_confirmed=ablation.get("confirmed"),
            ablation_status=ablation.get("status", "untested"),
            # Diagnostics
            diagnostics=data.get("diagnostics", {}),
            # Training diagnostics
            delta_alignment=delta_alignment,
            curvature=curvature,
            # Nonlinear geometry diagnostics
            nonlinear=nonlinear,
            probing=probing,
        )

    def __repr__(self) -> str:
        return (
            f"ComponentResult({self.component}, "
            f"C_unified={f'{self.C_unified:.3f}' if self.C_unified is not None else 'None'}, "
            f"paths={self.discovery_paths})"
        )


__all__ = ["ComponentResult"]
