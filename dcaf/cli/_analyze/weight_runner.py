"""
Weight domain analysis runner (sec:weight-analysis: C_W computation).

Loads saved deltas, computes per-parameter weight confidence (C_W) using
the projection-level confidence formula, and returns ranked candidates.

Usage:
    dcaf analyze -r ./runs/run_001/ --weight
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dcaf.arch.transformer import should_exclude_param
from dcaf.core.defaults import TAU_W_DEFAULT
from dcaf.core.signals import CANONICAL_SIGNALS
from dcaf.domains.weight import (
    compute_effectiveness_from_training_metrics,
    compute_opposition_degree,
    compute_projection_confidence,
    create_uniform_effectiveness,
    is_bidirectional,
)

logger = logging.getLogger(__name__)


def _classify_cluster(name: str) -> str:
    """Return '+', '-', or '0' for a delta name using canonical signal ids."""
    for sig in CANONICAL_SIGNALS:
        if name == f"delta_{sig.id}" or name.startswith(f"delta_{sig.id}_"):
            return sig.cluster
    n = name.lower()
    if "language" in n or "neutral" in n or "baseline" in n:
        return "0"
    if "target" in n or ("safe" in n and "anti" not in n and "negated" not in n):
        return "+"
    if "opposite" in n or ("adv" in n and "anti" not in n):
        return "-"
    return "+"


def run_weight_analysis(
    run_path: Path,
    tau_W: float = TAU_W_DEFAULT,
    top_k: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run weight domain analysis from CLI.

    Steps:
    1. Load deltas from DeltaStore
    2. Load training effectiveness metrics, falling back to uniform scores
    3. For each parameter: compute RMS norms, opposition degree, and C_W
    4. Filter by threshold and rank

    Args:
        run_path: Path to DCAF run directory
        tau_W: Weight confidence threshold
        top_k: Number of top candidates to return
        verbose: Show detailed output

    Returns:
        Analysis results dict with:
        - total_params: Total parameters analyzed
        - passing_threshold: Count above tau_W
        - top_candidates: List of top results
        - summary: Summary statistics
    """
    from dcaf.storage import DeltaStore

    logger.info("=" * 60)
    logger.info("WEIGHT DOMAIN ANALYSIS (C_W)")
    logger.info("=" * 60)

    # Load deltas
    delta_store = DeltaStore(run_path)
    available_deltas = delta_store.list_deltas()

    if not available_deltas:
        logger.error(f"No deltas found in {run_path}")
        return {"error": "No deltas found", "total_params": 0}

    logger.info(f"Found {len(available_deltas)} deltas:")
    for name in available_deltas:
        logger.info(f"  - {name}")

    # Load all deltas as {signal_name: {param_name: Tensor}}
    logger.info("\nLoading deltas...")
    deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in available_deltas:
        deltas[name] = delta_store.load_delta(name)
        if verbose:
            logger.info(f"  {name}: {len(deltas[name])} params")

    # Classify signals by cluster
    behavioral_signals = [n for n in available_deltas if _classify_cluster(n) != "0"]
    baseline_signals = [n for n in available_deltas if _classify_cluster(n) == "0"]
    t_plus = [n for n in available_deltas if _classify_cluster(n) == "+"]
    t_minus = [n for n in available_deltas if _classify_cluster(n) == "-"]

    logger.info("\nSignal clusters detected:")
    logger.info(f"  T+ (target):   {len(t_plus)} — {t_plus}")
    logger.info(f"  T- (opposite): {len(t_minus)} — {t_minus}")
    logger.info(f"  T0 (baseline): {len(baseline_signals)} — {baseline_signals}")

    # Load topology for projection-level analysis (§4: "Weight analysis
    # operates at projection level: each projection matrix is analyzed
    # individually, then results are aggregated to component level")
    from dcaf.core.topology import expand_deltas_to_projections
    from dcaf.domains.weight.delta import compute_projection_rms

    topo = delta_store.load_topology()
    logger.info(f"  Topology: {len(topo.projections)} projections, {len(topo.components)} components")

    proj_deltas_by_signal: Dict[str, Dict[str, torch.Tensor]] = {}
    rms_by_signal: Dict[str, Dict[str, float]] = {}
    for sig_name, delta_dict in deltas.items():
        proj_deltas = expand_deltas_to_projections(delta_dict, topo)
        proj_deltas_by_signal[sig_name] = proj_deltas
        rms_by_signal[sig_name] = {
            proj_id: compute_projection_rms(d)
            for proj_id, d in proj_deltas.items()
        }

    all_projs = sorted(set().union(*(d.keys() for d in rms_by_signal.values())))
    param_names = all_projs
    param_to_index = {name: idx for idx, name in enumerate(param_names)}

    def _agg_delta(signal_list: List[str], pname: str) -> Optional[torch.Tensor]:
        tensors = [
            proj_deltas_by_signal[s][pname]
            for s in signal_list
            if pname in proj_deltas_by_signal.get(s, {})
        ]
        if not tensors:
            return None
        return torch.stack(tensors).mean(0)

    # Effectiveness from training metrics when available; fall back explicitly.
    try:
        metadata = delta_store.load_metadata()
        training_metrics = metadata.extra.get("training_metrics", {})
    except Exception as exc:
        logger.warning("Could not load training metrics metadata: %s", exc)
        training_metrics = {}

    if training_metrics:
        effectiveness = compute_effectiveness_from_training_metrics(
            training_metrics,
            available_deltas,
        )
        missing = [name for name in available_deltas if name not in training_metrics]
        if missing:
            logger.warning(
                "Training metrics missing for %d deltas; default effectiveness used for: %s",
                len(missing),
                missing,
            )
    else:
        logger.warning("No training metrics found; using uniform effectiveness")
        effectiveness = create_uniform_effectiveness(available_deltas)

    # Score each parameter
    logger.info(f"\nScoring {len(param_names)} parameters...")

    results: Dict[str, Dict[str, Any]] = {}

    for pname in param_names:
        if should_exclude_param(pname):
            continue

        # Opposition degree
        tp_delta = _agg_delta(t_plus, pname)
        tm_delta = _agg_delta(t_minus, pname)
        opp_degree = 0.0
        if tp_delta is not None and tm_delta is not None:
            _, opp_degree = compute_opposition_degree(tp_delta, tm_delta)

        # C_W computation — pass FULL rms dict so percentile thresholds
        # are computed across ALL projections, not just this one
        c_w = compute_projection_confidence(
            proj=pname,
            rms_by_signal=rms_by_signal,
            effectiveness=effectiveness,
            opp_degree=opp_degree,
            behavioral_signals=behavioral_signals,
            baseline_signals=baseline_signals,
        )

        if c_w > 0:
            results[pname] = {
                "C_W": c_w,
                "opp_degree": opp_degree,
                "bidirectional": is_bidirectional(opp_degree),
                "signal_presence": sum(
                    1 for s in behavioral_signals if pname in rms_by_signal.get(s, {})
                ) / max(len(behavioral_signals), 1),
                "baseline_passed": c_w > 0,
                "contributing_signals": [
                    s for s in behavioral_signals if pname in rms_by_signal.get(s, {})
                ],
                "layer": _extract_layer(pname),
            }

    # Sort and filter
    sorted_results: List[Tuple[str, Dict]] = sorted(
        results.items(), key=lambda x: x[1]["C_W"], reverse=True
    )
    passing = [(n, r) for n, r in sorted_results if r["C_W"] >= tau_W]
    top_candidates = sorted_results[:top_k]

    # Summary statistics
    c_w_values = [r["C_W"] for _, r in sorted_results]
    summary = {
        "total_params": len(sorted_results),
        "mean_C_W": sum(c_w_values) / len(c_w_values) if c_w_values else 0.0,
        "max_C_W": max(c_w_values) if c_w_values else 0.0,
        "bidirectional_count": sum(1 for _, r in sorted_results if r["bidirectional"]),
        "baseline_passed_count": len(sorted_results),
    }

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {summary['total_params']}")
    logger.info(f"Passing tau_W={tau_W}: {len(passing)}")
    logger.info(f"Mean C_W: {summary['mean_C_W']:.4f}")
    logger.info(f"Max C_W: {summary['max_C_W']:.4f}")
    logger.info(f"Bidirectional params: {summary['bidirectional_count']}")

    if top_candidates:
        logger.info(f"\nTop {min(10, len(top_candidates))} candidates:")
        for i, (pname, r) in enumerate(top_candidates[:10]):
            short = pname if len(pname) <= 50 else pname[:47] + "..."
            logger.info(
                f"  {i+1:2}. C_W={r['C_W']:.4f} opp={r['opp_degree']:.3f} "
                f"bi={'Y' if r['bidirectional'] else 'N'} {short}"
            )

    def _candidate_record(pname: str, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "index": param_to_index[pname],
            "param_name": pname,
            "C_W": r["C_W"],
            "opp_degree": r["opp_degree"],
            "bidirectional": r["bidirectional"],
            "signal_presence": r["signal_presence"],
            "baseline_passed": r["baseline_passed"],
            "contributing_signals": r["contributing_signals"],
            "layer": r["layer"],
        }

    return {
        "total_params": summary["total_params"],
        "passing_threshold": len(passing),
        "threshold": tau_W,
        "summary": summary,
        "scores_by_param": {
            param_to_index[pname]: r["C_W"] for pname, r in sorted_results
        },
        "param_index_to_name": {
            param_to_index[pname]: pname for pname, _ in sorted_results
        },
        "all_candidates": [
            _candidate_record(pname, r) for pname, r in sorted_results
        ],
        "top_candidates": [
            _candidate_record(pname, r) for pname, r in top_candidates
        ],
    }


def _extract_layer(param_name: str) -> Optional[int]:
    """Extract layer index from a parameter name, or None if not a layer parameter."""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h", "layer") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def display_weight_results(
    results: Dict[str, Any],
    min_confidence: float = None,
    sort_by_confidence: bool = False,
) -> None:
    """Display weight analysis results in a formatted way."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("WEIGHT DOMAIN ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total parameters analyzed: {results['total_params']}")
    print(f"Passing threshold (tau_W={results['threshold']}): {results['passing_threshold']}")

    summary = results.get("summary", {})
    print("\nSummary:")
    print(f"  Mean C_W: {summary.get('mean_C_W', 0):.4f}")
    print(f"  Max C_W: {summary.get('max_C_W', 0):.4f}")
    print(f"  Bidirectional: {summary.get('bidirectional_count', 0)}")

    candidates = results.get("top_candidates", [])
    if min_confidence is not None:
        candidates = [c for c in candidates if c.get("C_W", 0) >= min_confidence]
    if sort_by_confidence:
        candidates = sorted(candidates, key=lambda c: c.get("C_W", 0), reverse=True)
    if candidates:
        print(f"\nTop {min(20, len(candidates))} Candidates:")
        print("-" * 80)
        for i, c in enumerate(candidates[:20]):
            name = c.get("param_name", f"idx_{c.get('index', '?')}")
            if name and len(name) > 45:
                name = name[:42] + "..."
            print(
                f"{i+1:3}. C_W={c['C_W']:.4f} | opp={c['opp_degree']:.3f} | "
                f"bi={'Y' if c['bidirectional'] else 'N'} | {name}"
            )


__all__ = ["run_weight_analysis", "display_weight_results"]
