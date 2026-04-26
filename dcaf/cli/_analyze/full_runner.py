"""
Full pipeline analysis runner (§12 Complete Pipeline: Theorem 12.1).

Orchestrates weight domain (C_W), activation domain (C_A), geometry domain (C_G),
triangulation, and optional circuit analysis.

Usage:
    dcaf analyze -r ./runs/run_001/
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

from dcaf.core.defaults import (
    TAU_W_DEFAULT,
    TAU_A_DEFAULT,
    TAU_G_DEFAULT,
    TOP_K_CANDIDATES,
)

logger = logging.getLogger(__name__)


def run_full_analysis(
    run_path: Path,
    model_name: Optional[str] = None,
    tau_W: float = TAU_W_DEFAULT,
    tau_A: float = TAU_A_DEFAULT,
    tau_G: float = TAU_G_DEFAULT,
    probe_size: int = 50,
    top_k: int = TOP_K_CANDIDATES,
    skip_activation: bool = False,
    skip_geometry: bool = False,
    skip_circuit: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full DCAF analysis pipeline.

    Steps:
    1. Weight domain analysis → C_W, H_W
    2. Activation domain analysis → C_A (optional)
    3. Geometry domain analysis → C_G (optional)
    4. Triangulate → H_cand
    5. Circuit analysis: Impact, Classification (optional)

    Args:
        run_path: Path to DCAF run directory
        model_name: Model name (auto-detected from metadata if None)
        tau_W: Weight confidence threshold
        tau_A: Activation confidence threshold
        tau_G: Geometry confidence threshold
        probe_size: Number of probe prompts
        top_k: Number of top candidates to return
        skip_activation: Skip activation domain
        skip_geometry: Skip geometry domain
        skip_circuit: Skip circuit analysis
        device: Device to use

    Returns:
        Complete analysis results dict
    """
    from dcaf.storage import DeltaStore
    from dcaf.candidates import (
        CandidateSet,
        CandidateInfo,
        CandidateStatus,
        create_discovery_set,
        create_validated_set,
    )
    from dcaf.confidence import (
        triangulate_batch,
        ThresholdConfig,
    )

    logger.info("=" * 60)
    logger.info("FULL DCAF ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # Load metadata
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()

    if model_name is None:
        if metadata and hasattr(metadata, 'model_name'):
            model_name = metadata.model_name
        else:
            logger.error("Could not determine model name")
            return {"error": "Model name required"}

    logger.info(f"Run: {run_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Thresholds: τ_W={tau_W}, τ_A={tau_A}, τ_G={tau_G}")

    results = {
        "run_path": str(run_path),
        "model_name": model_name,
        "thresholds": {"tau_W": tau_W, "tau_A": tau_A, "tau_G": tau_G},
    }

    # Check for pre-computed discovery
    discovery_path = run_path / "discovery.json"
    discovery_info = None
    H_disc = None
    use_precomputed_discovery = False

    if discovery_path.exists():
        logger.info("\n" + "-" * 60)
        logger.info("LOADING PRE-COMPUTED DISCOVERY")
        logger.info("-" * 60)
        logger.info(f"Found: {discovery_path}")

        from dcaf.cli._discover.integration import (
            load_discovery_result,
            discovery_result_to_sets,
        )

        discovery_data = load_discovery_result(discovery_path)
        H_disc, discovery_info, H_W_loaded, H_A_loaded, H_G_loaded = discovery_result_to_sets(discovery_data)
        use_precomputed_discovery = True

        results["discovery"] = {
            "source": "discovery.json",
            "H_disc_count": len(H_disc),
            "H_W_count": len(H_W_loaded),
            "H_A_count": len(H_A_loaded),
            "H_G_count": len(H_G_loaded),
            "summary": discovery_data.get("summary", {}),
        }

        logger.info(f"H_disc: {len(H_disc)} parameters from discovery.json")
        logger.info(f"  H_W: {len(H_W_loaded)}, H_A: {len(H_A_loaded)}, H_G: {len(H_G_loaded)}")
        logger.info(f"  Multi-path: {discovery_data.get('summary', {}).get('multi_path_count', 0)}")
    else:
        logger.info("\nNo discovery.json found - will compute discovery inline")
        results["discovery"] = {"source": "inline"}

    # ==========================================================================
    # PHASE 1: Weight Domain Analysis (C_W computation)
    # ==========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("PHASE 1: Weight Domain Analysis")
    logger.info("-" * 60)

    from dcaf.cli._analyze.weight_runner import run_weight_analysis

    # Always run weight analysis to get C_W confidence values
    weight_results = run_weight_analysis(
        run_path=run_path,
        tau_W=tau_W,
        top_k=top_k * 2,  # Get more candidates for filtering
        verbose=False,
    )

    results["weight"] = {
        "total_params": weight_results.get("total_params", 0),
        "passing_threshold": weight_results.get("passing_threshold", 0),
        "summary": weight_results.get("summary", {}),
    }

    logger.info(f"Weight analysis: {weight_results.get('passing_threshold', 0)} candidates passing τ_W")

    # Build confidence dict and param names from weight results
    threshold_config = ThresholdConfig(tau_W=tau_W, tau_A=tau_A, tau_G=tau_G)

    weight_confidences = {}
    param_names = {}
    for cand in weight_results.get("top_candidates", []):
        idx = cand["index"]
        weight_confidences[idx] = cand["C_W"]
        if cand.get("param_name"):
            param_names[idx] = cand["param_name"]

    # Create H_W from weight analysis (used when no discovery.json)
    H_W = create_discovery_set(
        weight_confidences=weight_confidences,
        threshold_config=threshold_config,
    )

    if use_precomputed_discovery:
        # H_disc loaded from discovery.json - report counts from there
        results["H_W_count"] = len(H_W_loaded)
        logger.info(f"H_W from discovery.json: {len(H_W_loaded)} params")
    else:
        results["H_W_count"] = len(H_W)
        logger.info(f"H_W (computed inline): {len(H_W)} candidates")

    # ==========================================================================
    # PHASE 2: Activation Domain Analysis (Optional)
    # ==========================================================================
    activation_confidences = {}

    if not skip_activation:
        logger.info("\n" + "-" * 60)
        logger.info("PHASE 2: Activation Domain Analysis")
        logger.info("-" * 60)

        try:
            from dcaf.cli._analyze.activation_runner import run_activation_analysis

            activation_results = run_activation_analysis(
                run_path=run_path,
                model_name=model_name,
                tau_A=tau_A,
                probe_size=probe_size,
                top_k=100,
                device=device,
            )

            results["activation"] = {
                "total_components": activation_results.get("total_components", 0),
                "passing_threshold": activation_results.get("passing_threshold", 0),
                "summary": activation_results.get("summary", {}),
            }

            # Build activation confidence dict by component
            for comp_data in activation_results.get("top_components", []):
                activation_confidences[comp_data["component"]] = comp_data["C_A"]

            logger.info(f"Activation analysis: {activation_results.get('passing_threshold', 0)} components passing τ_A")

        except Exception as e:
            logger.warning(f"Activation analysis failed: {e}")
            results["activation"] = {"error": str(e)}
    else:
        logger.info("\nSkipping activation domain analysis (--skip-activation)")
        results["activation"] = {"skipped": True}

    # ==========================================================================
    # PHASE 3: Geometry Domain Analysis (Optional)
    # ==========================================================================
    geometry_confidences = {}

    if not skip_geometry:
        logger.info("\n" + "-" * 60)
        logger.info("PHASE 3: Geometry Domain Analysis")
        logger.info("-" * 60)

        try:
            from dcaf.cli._analyze.geometry_runner import run_geometry_analysis

            geometry_results = run_geometry_analysis(
                run_path=run_path,
                model_name=model_name,
                tau_G=tau_G,
                probe_size=probe_size,
                top_k=100,
                device=device,
            )

            results["geometry"] = {
                "total_components": geometry_results.get("total_components", 0),
                "passing_threshold": geometry_results.get("passing_threshold", 0),
                "summary": geometry_results.get("summary", {}),
            }

            # Build geometry confidence dict by component
            for comp_data in geometry_results.get("top_components", []):
                geometry_confidences[comp_data["component"]] = comp_data["C_G"]

            logger.info(f"Geometry analysis: {geometry_results.get('passing_threshold', 0)} components passing τ_G")

        except Exception as e:
            logger.warning(f"Geometry analysis failed: {e}")
            results["geometry"] = {"error": str(e)}
    else:
        logger.info("\nSkipping geometry domain analysis (--skip-geometry)")
        results["geometry"] = {"skipped": True}

    # ==========================================================================
    # PHASE 4: Triangulation and Candidate Filtering
    # ==========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("PHASE 4: Triangulation & Candidate Filtering")
    logger.info("-" * 60)

    # Determine base discovery set
    # When pre-computed discovery exists, use H_disc (union of all paths)
    # Otherwise use H_W computed inline
    base_discovery_set = H_W
    if use_precomputed_discovery and H_disc:
        # Create discovery set from H_disc with weight confidences
        base_discovery_set = create_discovery_set(
            weight_confidences={idx: weight_confidences.get(idx, 0.0) for idx in H_disc},
            threshold_config=threshold_config,
        )
        logger.info(f"Using H_disc from discovery.json: {len(H_disc)} params")

    # Create H_cand by filtering through activation and geometry
    H_cand = create_validated_set(
        discovery_set=base_discovery_set,
        activation_confidences=activation_confidences,
        geometry_confidences=geometry_confidences,
        threshold_config=threshold_config,
    )

    results["H_cand_count"] = len(H_cand)
    logger.info(f"H_cand (validated set): {len(H_cand)} candidates")

    # Triangulate confidences
    triangulation_input = {}
    for cid, info in H_cand.candidates.items():
        triangulation_input[cid] = {
            "C_W": info.C_W,
            "C_A": info.C_A,
            "C_G": info.C_G,
        }

    triangulated = triangulate_batch(triangulation_input)

    # Apply multi-path bonus if discovery_info available
    unified_scores = {}
    for cid, tri in triangulated.items():
        C_base = tri.value
        bonus = 0.0

        # Get bonus from discovery_info if available
        if discovery_info and cid in discovery_info:
            bonus = discovery_info[cid].bonus

        # C_unified = min(1, C_base + bonus)
        unified_scores[cid] = min(1.0, C_base + bonus)

    results["triangulated_count"] = len(triangulated)

    # Get top by unified confidence (with bonus applied)
    top_triangulated = sorted(
        unified_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results["top_candidates"] = [
        {
            "id": cid,
            "param_name": param_names.get(cid),
            "C_unified": score,
            "C_base": triangulated[cid].value if cid in triangulated else None,
            "bonus": discovery_info[cid].bonus if discovery_info and cid in discovery_info else 0.0,
            "paths": discovery_info[cid].paths if discovery_info and cid in discovery_info else [],
            "C_W": H_cand.candidates[cid].C_W if cid in H_cand.candidates else None,
            "C_A": H_cand.candidates[cid].C_A if cid in H_cand.candidates else None,
            "C_G": H_cand.candidates[cid].C_G if cid in H_cand.candidates else None,
        }
        for cid, score in top_triangulated
    ]

    # ==========================================================================
    # STEP 5: Circuit Analysis (Optional)
    # ==========================================================================
    if not skip_circuit and len(H_cand) > 0:
        logger.info("\n" + "-" * 60)
        logger.info("STEP 5: Circuit Analysis")
        logger.info("-" * 60)

        # Get unique components from candidates
        components = set()
        for info in H_cand.candidates.values():
            if info.component:
                components.add(info.component)

        logger.info(f"Components in H_cand: {len(components)}")

        # Note: Full circuit analysis (impact, classification, edges, steering)
        # requires model loading and is expensive. For now, just report.
        results["circuit"] = {
            "components_identified": len(components),
            "status": "summary_only",
            "note": "Full circuit analysis requires additional compute",
        }
    else:
        results["circuit"] = {"skipped": True}

    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"H_W (discovery): {results['H_W_count']} params")
    logger.info(f"H_cand (validated): {results['H_cand_count']} params")
    logger.info(f"Top triangulated: {len(results['top_candidates'])} params")

    if results["top_candidates"]:
        logger.info(f"\nTop 5 candidates by unified confidence:")
        for i, c in enumerate(results["top_candidates"][:5]):
            name = c.get("param_name", f"idx_{c['id']}")
            if name and len(name) > 40:
                name = name[:37] + "..."
            bonus = c.get("bonus", 0)
            paths = c.get("paths", [])
            path_str = f" +{bonus:.2f}" if bonus > 0 else ""
            logger.info(
                f"  {i+1}. C={c['C_unified']:.4f}{path_str} "
                f"(W={c.get('C_W', 0) or 0:.3f}, A={c.get('C_A', 0) or 0:.3f}, G={c.get('C_G', 0) or 0:.3f}) "
                f"{name}"
            )

    return results


def display_full_results(results: Dict[str, Any]) -> None:
    """Display full analysis results in a formatted way."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "=" * 70)
    print("FULL DCAF ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Run: {results.get('run_path')}")
    print(f"Model: {results.get('model_name')}")

    thresholds = results.get("thresholds", {})
    print(f"Thresholds: τ_W={thresholds.get('tau_W')}, τ_A={thresholds.get('tau_A')}, τ_G={thresholds.get('tau_G')}")

    print("\n" + "-" * 70)
    print("DOMAIN SUMMARY")
    print("-" * 70)

    weight = results.get("weight", {})
    print(f"Weight:     {weight.get('passing_threshold', 0):4} / {weight.get('total_params', 0)} params passing")

    activation = results.get("activation", {})
    if activation.get("skipped"):
        print("Activation: SKIPPED")
    elif activation.get("error"):
        print(f"Activation: ERROR - {activation.get('error')}")
    else:
        print(f"Activation: {activation.get('passing_threshold', 0):4} / {activation.get('total_components', 0)} components passing")

    geometry = results.get("geometry", {})
    if geometry.get("skipped"):
        print("Geometry:   SKIPPED")
    elif geometry.get("error"):
        print(f"Geometry:   ERROR - {geometry.get('error')}")
    else:
        print(f"Geometry:   {geometry.get('passing_threshold', 0):4} / {geometry.get('total_components', 0)} components passing")

    print("\n" + "-" * 70)
    print("CANDIDATE PIPELINE")
    print("-" * 70)
    print(f"H_W (discovery):     {results.get('H_W_count', 0):4} params")
    print(f"H_cand (validated):  {results.get('H_cand_count', 0):4} params")
    print(f"Triangulated:        {results.get('triangulated_count', 0):4} params")

    # Show discovery source
    discovery = results.get("discovery", {})
    if discovery.get("source") == "discovery.json":
        print(f"\nDiscovery: Loaded from discovery.json ({discovery.get('H_disc_count', 0)} params)")
        print(f"  Multi-path: {discovery.get('summary', {}).get('multi_path_count', 0)} params")

    candidates = results.get("top_candidates", [])
    if candidates:
        print("\n" + "-" * 70)
        print("TOP CANDIDATES (by unified confidence)")
        print("-" * 70)
        print(f"{'#':>3} | {'C_unified':>8} | {'bonus':>5} | {'C_W':>6} | {'C_A':>6} | {'C_G':>6} | Param Name")
        print("-" * 70)
        for i, c in enumerate(candidates[:20]):
            name = c.get("param_name", f"idx_{c['id']}")
            if name and len(name) > 30:
                name = name[:27] + "..."
            cw = c.get("C_W") or 0
            ca = c.get("C_A") or 0
            cg = c.get("C_G") or 0
            bonus = c.get("bonus") or 0
            print(f"{i+1:3} | {c['C_unified']:8.4f} | {bonus:5.2f} | {cw:6.3f} | {ca:6.3f} | {cg:6.3f} | {name}")


__all__ = ["run_full_analysis", "display_full_results"]
