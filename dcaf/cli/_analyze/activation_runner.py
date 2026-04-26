"""
Activation domain analysis runner (§5 Activation Analysis: C_A computation).

Loads model, applies each saved delta, captures activations, and computes
per-component activation confidence (C_A).

Usage:
    dcaf analyze -r ./runs/run_001/ --activation
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import torch

logger = logging.getLogger(__name__)


def run_activation_analysis(
    run_path: Path,
    model_name: Optional[str] = None,
    tau_A: float = 0.3,
    probe_size: int = 50,
    probe_type: str = "both",
    top_k: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run activation domain analysis from CLI.

    Steps:
    1. Load model and checkpoints
    2. For each available delta (signal):
       - Apply the delta
       - Capture activations
       - Compute magnitudes vs baseline
    3. Compute C_A for all components
    4. Filter and rank

    Args:
        run_path: Path to DCAF run directory
        model_name: Model name (auto-detected from metadata if None)
        tau_A: Activation confidence threshold
        probe_size: Number of probe prompts
        probe_type: Type of probes ("recognition", "generation", "both")
        top_k: Number of top components to return
        device: Device to use

    Returns:
        Analysis results dict
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dcaf.storage import DeltaStore
    from dcaf.ablation import ModelStateManager
    from dcaf.domains.activation import (
        ActivationCapture,
        ProbeSet,
        compute_magnitude_from_snapshots,
        compute_all_activation_confidences,
        rank_by_activation_confidence,
        get_confidence_summary,
    )

    logger.info("=" * 60)
    logger.info("ACTIVATION DOMAIN ANALYSIS (C_A)")
    logger.info("=" * 60)

    # Load delta store and metadata
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()

    if model_name is None:
        if metadata and hasattr(metadata, 'model_name'):
            model_name = metadata.model_name
        else:
            logger.error("Could not determine model name. Provide --model-name")
            return {"error": "Model name required"}

    logger.info(f"Model: {model_name}")

    # List available deltas
    available_deltas = delta_store.list_deltas()
    if not available_deltas:
        logger.error(f"No deltas found in {run_path}")
        return {"error": "No deltas found"}

    logger.info(f"Found {len(available_deltas)} deltas: {available_deltas}")

    # Load base checkpoint
    base_checkpoint = delta_store.load_checkpoint("base")
    if not base_checkpoint:
        logger.error("No base checkpoint found")
        return {"error": "No base checkpoint"}

    # Create probe set
    category = None
    if metadata and hasattr(metadata, 'dataset_config'):
        category = metadata.dataset_config.get("category")

    if category:
        logger.info(f"Creating probe set for category: {category}")
        probe_set = ProbeSet.from_category(category, size=probe_size)
    else:
        logger.info("Creating default probe set")
        probe_set = ProbeSet.default(size=probe_size)

    logger.info(f"Probe set: {len(probe_set)} prompts")

    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")

    # Load model
    logger.info(f"\nLoading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply base checkpoint
    logger.info("Applying base checkpoint...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_checkpoint:
                param.copy_(base_checkpoint[name].to(device))

    # Initialize activation capture
    capture = ActivationCapture(model)

    # Capture baseline activations (base model)
    logger.info("\nCapturing baseline activations (base model)...")
    baseline_snapshot = capture.capture(
        probe_set, tokenizer, name="baseline", probe_type=probe_type
    )

    # Get all component names
    all_components: Set[str] = set()
    all_components.update(baseline_snapshot.attention_patterns.keys())
    all_components.update(baseline_snapshot.mlp_activations.keys())

    logger.info(f"  Components captured: {len(all_components)}")

    # For each delta, capture activations and compute magnitudes
    magnitudes_by_signal_probe: Dict[tuple, Dict[str, float]] = {}

    # Find primary target-side delta for state manager
    safety_delta_name = next(
        (d for d in available_deltas if "t1_prefopt_target" in d),
        next((d for d in available_deltas if "target" in d),
             next((d for d in available_deltas if "safe" in d), available_deltas[0]))
    )
    safety_delta = delta_store.load_delta(safety_delta_name)

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta, delta_scale=1.0
    )

    for signal_name in available_deltas:
        logger.info(f"\nProcessing signal: {signal_name}")

        # Load and apply this delta
        delta = delta_store.load_delta(signal_name)

        # Reset to base and apply this specific delta
        state_manager.reset_to_base()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in delta:
                    param.add_(delta[name].to(device))

        # Capture activations after this delta
        signal_snapshot = capture.capture(
            probe_set, tokenizer, name=signal_name, probe_type=probe_type
        )

        # Compute magnitudes (comparing to baseline)
        # Combine attention and MLP into single dict
        post_acts = {}
        pre_acts = {}

        for comp in signal_snapshot.attention_patterns.keys():
            post_acts[comp] = signal_snapshot.attention_patterns[comp]
            if comp in baseline_snapshot.attention_patterns:
                pre_acts[comp] = baseline_snapshot.attention_patterns[comp]

        for comp in signal_snapshot.mlp_activations.keys():
            post_acts[comp] = signal_snapshot.mlp_activations[comp]
            if comp in baseline_snapshot.mlp_activations:
                pre_acts[comp] = baseline_snapshot.mlp_activations[comp]

        magnitudes = compute_magnitude_from_snapshots(post_acts, pre_acts)

        # Store by (signal, probe_type) key
        # For now use single probe_type
        magnitudes_by_signal_probe[(signal_name, probe_type)] = magnitudes

        logger.info(f"  Computed magnitudes for {len(magnitudes)} components")

    # Compute C_A for all components
    logger.info(f"\nComputing C_A for {len(all_components)} components...")

    results = compute_all_activation_confidences(
        components=all_components,
        magnitudes_by_signal_probe=magnitudes_by_signal_probe,
        tau_act=85.0,  # Default percentile
    )

    # Filter and rank
    ranked = rank_by_activation_confidence(results)
    summary = get_confidence_summary(results)

    # Count passing threshold
    passing_count = sum(1 for r in results.values() if r.C_A >= tau_A)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total components: {summary['count']}")
    logger.info(f"Passing τ_A={tau_A}: {passing_count}")
    logger.info(f"Mean C_A: {summary.get('mean_C_A', 0):.4f}")
    logger.info(f"Max C_A: {summary.get('max_C_A', 0):.4f}")
    logger.info(f"Significant (n_A > 0): {summary.get('significant_count', 0)}")

    if ranked:
        logger.info(f"\nTop {min(10, len(ranked))} components:")
        for i, (component, result) in enumerate(ranked[:10]):
            logger.info(
                f"  {i+1:2}. C_A={result.C_A:.4f} n_A={result.n_A}/{result.total_pairs} {component}"
            )

    # Cleanup
    del model, capture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "total_components": summary["count"],
        "passing_threshold": passing_count,
        "threshold": tau_A,
        "summary": summary,
        "top_components": [
            {
                "component": comp,
                "C_A": result.C_A,
                "n_A": result.n_A,
                "total_pairs": result.total_pairs,
                "significant_pairs": result.significant_pairs,
            }
            for comp, result in ranked[:top_k]
        ],
    }


def display_activation_results(results: Dict[str, Any]) -> None:
    """Display activation analysis results in a formatted way."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("ACTIVATION DOMAIN ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total components analyzed: {results['total_components']}")
    print(f"Passing threshold (τ_A={results['threshold']}): {results['passing_threshold']}")

    summary = results.get("summary", {})
    print(f"\nSummary:")
    print(f"  Mean C_A: {summary.get('mean_C_A', 0):.4f}")
    print(f"  Max C_A: {summary.get('max_C_A', 0):.4f}")
    print(f"  Significant components: {summary.get('significant_count', 0)}")

    components = results.get("top_components", [])
    if components:
        print(f"\nTop {min(20, len(components))} Components:")
        print("-" * 60)
        for i, c in enumerate(components[:20]):
            print(
                f"{i+1:3}. C_A={c['C_A']:.4f} | n_A={c['n_A']:2}/{c['total_pairs']:2} | {c['component']}"
            )


__all__ = ["run_activation_analysis", "display_activation_results"]
