"""
Full-run output JSON schema assembly (§13, Def 13.1-13.4).

Assembles all DCAF analysis results into a unified JSON output schema.
Covers the complete pipeline output: discovery summary, domain summaries,
triangulated confidence, per-component results, and ablation confirmation.

For the lightweight in-memory per-component assembler, see dcaf.output.results.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

import dcaf


def assemble_output(
    run_path: str,
    model_name: str,
    variant_name: str,
    discovery_summary: Dict[str, Any],
    weight_summary: Dict[str, Any],
    activation_summary: Optional[Dict[str, Any]] = None,
    geometry_summary: Optional[Dict[str, Any]] = None,
    triangulation_summary: Optional[Dict[str, Any]] = None,
    component_results: Optional[List[Dict[str, Any]]] = None,
    ablation_summary: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble complete DCAF output matching Spec Section 13.

    Args:
        run_path: Path to the run directory
        model_name: Name of the model analyzed
        variant_name: Training variant name
        discovery_summary: Discovery path summary (H_W, H_A, H_G counts)
        weight_summary: Weight domain analysis summary
        activation_summary: Activation domain analysis summary (optional)
        geometry_summary: Geometry domain analysis summary (optional)
        triangulation_summary: Triangulated confidence summary (optional)
        component_results: List of ComponentResult.to_dict() outputs
        ablation_summary: Ablation testing summary (optional)
        thresholds: Threshold configuration used
        config: Full configuration used

    Returns:
        Dict matching spec Section 13 output schema:
        {
            "dcaf_version", "timestamp", "run",
            "discovery", "domains": {weight, activation, geometry},
            "triangulation", "components": [ComponentResult.to_dict()],
            "ablation", "config"
        }
    """
    return {
        "dcaf_version": dcaf.__version__,
        "timestamp": datetime.now().isoformat(),
        "run": {
            "path": run_path,
            "model": model_name,
            "variant": variant_name,
        },
        "discovery": discovery_summary,
        "domains": {
            "weight": weight_summary,
            "activation": activation_summary or {},
            "geometry": geometry_summary or {},
        },
        "triangulation": triangulation_summary or {},
        "components": component_results or [],
        "ablation": ablation_summary or {},
        "thresholds": thresholds or {},
        "config": config or {},
    }


def assemble_component_output(
    component: str,
    param_names: List[str],
    scores: Dict[str, Optional[float]],
    discovery: Dict[str, Any],
    weight_details: Optional[Dict[str, Any]] = None,
    geometry_details: Optional[Dict[str, Any]] = None,
    classification: Optional[Dict[str, Any]] = None,
    interaction: Optional[Dict[str, Any]] = None,
    ablation: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble output for a single component (matches ComponentResult.to_dict()).

    Args:
        component: Component ID
        param_names: Parameter names in this component
        scores: {"C_W": val, "C_A": val, "C_G": val, "C_unified": val}
        discovery: {"paths": [...], "path_count": n, "multi_path_bonus": val}
        weight_details: {"bidirectional": bool, "opp_degree": float}
        geometry_details: {"lrs": float, "lrs_breakdown": {...}}
        classification: Tiered classification result
        interaction: {"requirement": str, "partners": [...], "type": str}
        ablation: {"confirmed": bool, "status": str}
        diagnostics: {"contributions": {}, "deviations": {}, "disagreement": float}

    Returns:
        Dict matching ComponentResult.to_dict() format
    """
    return {
        "component": component,
        "param_names": param_names,
        "scores": scores,
        "discovery": discovery,
        "weight_details": weight_details or {"bidirectional": False, "opp_degree": 0.0},
        "geometry_details": geometry_details or {"lrs": None, "lrs_breakdown": None},
        "classification": classification,
        "interaction": interaction or {"requirement": None, "partners": [], "type": None},
        "ablation": ablation or {"confirmed": None, "status": "untested"},
        "diagnostics": diagnostics or {},
    }


def assemble_discovery_summary(
    H_W: int,
    H_A: int,
    H_G: int,
    H_disc: int,
    H_cand: int,
    H_conf: int,
    multi_path_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Assemble discovery path summary.

    Args:
        H_W: Count of weight-discovered parameters
        H_A: Count of activation-discovered parameters
        H_G: Count of gradient-discovered parameters
        H_disc: Count of union (any path)
        H_cand: Count of candidates (passed threshold)
        H_conf: Count of confirmed (passed ablation)
        multi_path_stats: {"single": n, "double": n, "triple": n}

    Returns:
        Discovery summary dict
    """
    return {
        "paths": {
            "H_W": H_W,
            "H_A": H_A,
            "H_G": H_G,
        },
        "H_disc": H_disc,
        "H_cand": H_cand,
        "H_conf": H_conf,
        "multi_path": multi_path_stats or {},
    }


def assemble_domain_summary(
    domain: str,
    param_count: int,
    mean_confidence: float,
    median_confidence: float,
    above_threshold: int,
    threshold: float,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble summary for a single domain.

    Args:
        domain: "weight", "activation", or "geometry"
        param_count: Number of parameters analyzed
        mean_confidence: Mean confidence score
        median_confidence: Median confidence score
        above_threshold: Count above threshold
        threshold: Threshold used
        details: Domain-specific details

    Returns:
        Domain summary dict
    """
    return {
        "domain": domain,
        "param_count": param_count,
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "above_threshold": above_threshold,
        "threshold": threshold,
        "details": details or {},
    }


def validate_output(output: Dict[str, Any]) -> List[str]:
    """
    Validate output schema completeness.

    Args:
        output: Output dict to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    warnings = []

    # Required top-level keys
    required = ["dcaf_version", "timestamp", "run", "discovery", "domains", "components"]
    for key in required:
        if key not in output:
            warnings.append(f"Missing required key: {key}")

    # Check run info
    run = output.get("run", {})
    for key in ["path", "model", "variant"]:
        if key not in run:
            warnings.append(f"Missing run.{key}")

    # Check domains
    domains = output.get("domains", {})
    if "weight" not in domains:
        warnings.append("Missing domains.weight (required)")

    # Check components
    components = output.get("components", [])
    if not components:
        warnings.append("No components in output")

    for i, comp in enumerate(components):
        if "component" not in comp:
            warnings.append(f"Component {i} missing 'component' field")
        if "scores" not in comp:
            warnings.append(f"Component {i} missing 'scores' field")

    return warnings


__all__ = [
    "assemble_output",
    "assemble_component_output",
    "assemble_discovery_summary",
    "assemble_domain_summary",
    "validate_output",
]
