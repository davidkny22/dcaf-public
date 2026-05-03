"""
Probe analysis runner for DCAF (sec:circuit-graph).

Captures pre- and post-training activations, identifies circuits from weight
candidates, and validates circuits with safety measurements.

All activation capture happens during analysis — no special training flags needed.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dcaf.core.defaults import ATTENTION_WEIGHT, TAU_EDGE

logger = logging.getLogger(__name__)


def run_probe_analysis(
    run_path: Path,
    matching_params: List[str],
    model_name: str,
    cluster_method: str = "disjoint",
    edge_threshold: float = TAU_EDGE,
    attention_weight: float = ATTENTION_WEIGHT,
    probe_type: str = "both",
    probe_size: int = 100,
    category: Optional[str] = None,
    weight_classifications: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run probe analysis on DCAF-identified weight candidates.

    Captures activations and identifies circuits from weight candidates.
    All activation capture happens during analysis - works on any training run.

    Args:
        run_path: Path to DCAF run directory
        matching_params: DCAF-identified weight parameters
        model_name: Model name for loading
        cluster_method: Clustering method for circuits
        edge_threshold: Threshold for edge inclusion
        attention_weight: Weight factor for attention edges
        probe_type: Type of probes to capture ("recognition", "generation", or "both")
        probe_size: Number of probe prompts (default: 100)
        category: Harm category for probes (default: training run's category, or all)
        weight_classifications: Optional weight classifications from multi-probe ablation
        device: Device to use

    Returns:
        Circuit analysis results dict, or None if unavailable
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not matching_params:
        logger.warning("No matching params to analyze")
        return None

    try:
        from dcaf.circuit import CircuitIdentifier
        from dcaf.domains.activation import (
            ActivationCapture,
            ActivationDelta,
            ProbeSet,
        )
    except ImportError as e:
        logger.error(f"Probing/circuit module not available: {e}")
        return None

    logger.info("=" * 60)
    logger.info("PROBE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Weight candidates: {len(matching_params)}")
    logger.info(f"Clustering method: {cluster_method}")
    logger.info(f"Edge threshold: {edge_threshold}")
    logger.info(f"Attention weight: {attention_weight}")
    logger.info("")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load delta store and metadata
    from dcaf.storage.delta_store import DeltaStore
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()

    # Determine effective category: explicit > training run's category > all
    training_category = None
    if metadata and hasattr(metadata, 'dataset_config'):
        training_category = metadata.dataset_config.get("category")

    effective_category = category or training_category

    # Create probe set
    if effective_category:
        logger.info(f"Creating category-specific probe set: {effective_category}")
        probe_set = ProbeSet.from_category(effective_category, size=probe_size)
    else:
        logger.info("Creating all-category probe set")
        probe_set = ProbeSet.default(size=probe_size)
    logger.info(f"  Probe set: {probe_set.name} ({len(probe_set)} prompts)")

    # Load model
    logger.info("\nLoading model for activation capture...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoints
    base_checkpoint = delta_store.load_checkpoint("base")

    # Find primary target-side delta dynamically
    available_deltas = delta_store.list_deltas()
    safety_delta_name = next(
        (d for d in available_deltas if "t1_prefopt_target" in d),
        next((d for d in available_deltas if "t2_sft_target" in d),
             next((d for d in available_deltas
                   if "target" in d and "anti" not in d and "negated" not in d),
             next((d for d in available_deltas if "safe" in d), None))
    )
    )
    if not safety_delta_name:
        logger.error(f"No target delta found. Available: {available_deltas}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": "No target delta found",
            "available_deltas": available_deltas,
        }

    logger.info(f"Using safety delta: {safety_delta_name}")
    safety_delta = delta_store.load_delta(safety_delta_name)

    # Apply base checkpoint (pre-training state)
    logger.info("Applying base checkpoint to model...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_checkpoint:
                param.copy_(base_checkpoint[name].to(device))

    # Initialize activation capture
    capture = ActivationCapture(model)

    # Capture pre-training activations (base model state)
    logger.info("\nCapturing pre-training activations (base model state)...")
    pre_snapshot = capture.capture(
        probe_set,
        tokenizer,
        name="pre_training",
        probe_type=probe_type,
    )
    logger.info(f"  Attention heads: {len(pre_snapshot.attention_patterns)}")
    logger.info(f"  MLP components: {len(pre_snapshot.mlp_activations)}")

    # Apply safety delta (post-training state)
    logger.info("\nApplying safety delta to model...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in safety_delta:
                param.add_(safety_delta[name].to(device))

    # Capture post-training activations
    logger.info("Capturing post-training activations...")
    post_snapshot = capture.capture(
        probe_set,
        tokenizer,
        name="post_training",
        probe_type=probe_type,
    )
    logger.info(f"  Attention heads: {len(post_snapshot.attention_patterns)}")
    logger.info(f"  MLP components: {len(post_snapshot.mlp_activations)}")

    # Compute training delta
    training_delta = ActivationDelta(pre_snapshot, post_snapshot, "training")

    # Compute probe_activations for probe-response clustering
    logger.info("\nComputing probe activation patterns...")
    probe_activations: Dict[str, Dict[str, float]] = {}

    for probe_idx, prompt in enumerate(probe_set.all_prompts):
        probe_activations[prompt] = {}

        # MLP activations
        for component, tensor in post_snapshot.mlp_activations.items():
            # tensor shape: [batch, seq, hidden] or [batch, seq]
            # Mean over sequence dimension for this specific probe
            activation_value = tensor[probe_idx].mean().item()
            probe_activations[prompt][component] = activation_value

        # Attention patterns
        for component, tensor in post_snapshot.attention_patterns.items():
            # tensor shape: [batch, seq, seq]
            # Mean over all dimensions for this probe
            activation_value = tensor[probe_idx].mean().item()
            probe_activations[prompt][component] = activation_value

    logger.info(f"  Computed {len(probe_activations)} probe × "
                f"{len(probe_activations[list(probe_activations.keys())[0]] if probe_activations else 0)} component activations")

    # Compute harmful vs neutral activations for functional clustering
    logger.info("Computing harmful/neutral activation patterns...")
    harmful_activations: Dict[str, float] = {}
    neutral_activations: Dict[str, float] = {}

    # Split by probe type
    harmful_indices = [i for i, p in enumerate(probe_set.all_prompts) if p in probe_set.harmful_prompts]
    neutral_indices = [i for i, p in enumerate(probe_set.all_prompts) if p in probe_set.neutral_prompts]

    # MLP activations
    for component, tensor in post_snapshot.mlp_activations.items():
        # Mean over harmful probes
        if len(harmful_indices) > 0:
            harmful_tensor = tensor[harmful_indices]
            harmful_activations[component] = harmful_tensor.mean().item()

        # Mean over neutral probes
        if len(neutral_indices) > 0:
            neutral_tensor = tensor[neutral_indices]
            neutral_activations[component] = neutral_tensor.mean().item()

    # Attention patterns
    for component, tensor in post_snapshot.attention_patterns.items():
        if len(harmful_indices) > 0:
            harmful_tensor = tensor[harmful_indices]
            harmful_activations[component] = harmful_tensor.mean().item()

        if len(neutral_indices) > 0:
            neutral_tensor = tensor[neutral_indices]
            neutral_activations[component] = neutral_tensor.mean().item()

    logger.info(f"  Harmful components: {len(harmful_activations)}")
    logger.info(f"  Neutral components: {len(neutral_activations)}")

    # Capture post-ablation activations for each weight candidate
    logger.info(f"\nCapturing post-ablation activations ({len(matching_params)} weights)...")
    ablation_deltas: Dict[str, ActivationDelta] = {}

    # Create state manager for ablation
    from dcaf.ablation import ModelStateManager

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta, delta_scale=1.0
    )

    # Limit to top weights by delta magnitude for efficiency
    param_magnitudes = []
    for param_name in matching_params:
        if param_name in safety_delta:
            mag = torch.norm(safety_delta[param_name]).item()
            param_magnitudes.append((param_name, mag))
    param_magnitudes.sort(key=lambda x: x[1], reverse=True)
    top_params = [p[0] for p in param_magnitudes[:min(30, len(param_magnitudes))]]

    logger.info(f"  Analyzing top {len(top_params)} weights by magnitude...")

    for i, param_name in enumerate(top_params):
        logger.info(f"  [{i+1}/{len(top_params)}] {param_name[:50]}...")

        # Ablate this parameter
        state_manager.reset_to_safety()
        state_manager.ablate_params([param_name])

        # Capture activations
        ablated_snapshot = capture.capture(
            probe_set, tokenizer, name=f"post_ablation_{i}"
        )

        # Compute ablation delta (compared to post-training)
        ablation_delta = ActivationDelta(post_snapshot, ablated_snapshot, "ablation")
        ablation_deltas[param_name] = ablation_delta

        # Restore
        state_manager.reset_to_safety()

    # Save activations for potential reuse
    analysis_dir = run_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    pre_snapshot.save(str(analysis_dir / "pre_activations.pt"))
    post_snapshot.save(str(analysis_dir / "post_activations.pt"))
    logger.info(f"\nActivations saved to: {analysis_dir}")

    # Create safety measurement function for circuit validation
    logger.info("Initializing refusal classifier for circuit validation...")
    from dcaf.evaluation.refusal import RefusalClassifier

    classifier = RefusalClassifier()

    def measure_safety_fn(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        probe_set=probe_set,
    ) -> float:
        """Measure safety impact by running harmful prompts and counting refusals."""
        refusal_count = 0
        total_count = 0

        for prompt in probe_set.harmful_prompts:
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Classify response
            result = classifier.classify(response, prompt)

            # Count refusals (REFUSE + AVOID)
            if result.is_refusal:
                refusal_count += 1
            total_count += 1

        # Return refusal rate as safety metric
        return refusal_count / total_count if total_count > 0 else 0.0

    # Run circuit identification
    logger.info("\nIdentifying circuits...")
    identifier = CircuitIdentifier(
        training_delta=training_delta,
        ablation_deltas=ablation_deltas,
        weight_candidates=matching_params,
        probe_set=probe_set,
        edge_threshold=edge_threshold,
        attention_weight=attention_weight,
    )

    circuit_results = identifier.identify_circuits(
        method=cluster_method,
        state_manager=state_manager,
        measure_safety_fn=measure_safety_fn,
        probe_activations=probe_activations,
        harmful_activations=harmful_activations,
        neutral_activations=neutral_activations,
    )

    # Add weight classifications to circuits if available
    if weight_classifications:
        logger.info("\nAdding weight classifications to circuits...")
        identifier.add_weight_classifications(
            circuit_results.circuits,
            weight_classifications,
        )
        # Log circuit type distribution
        circuit_types = {}
        for circuit in circuit_results.circuits:
            ct = circuit.circuit_type or "mixed"
            circuit_types[ct] = circuit_types.get(ct, 0) + 1
        logger.info("Circuit types:")
        for ct, count in sorted(circuit_types.items()):
            logger.info(f"  {ct}: {count}")

    # Save circuits
    circuits_path = analysis_dir / "circuits.json"
    circuit_results.save(str(circuits_path))
    logger.info(f"\nCircuits saved to: {circuits_path}")

    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("CIRCUIT ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Circuits identified: {len(circuit_results.circuits)}")
    logger.info(f"Clustering method: {circuit_results.clustering_method}")
    logger.info(f"Edge threshold: {circuit_results.edge_threshold}")
    logger.info("")

    for i, circuit in enumerate(circuit_results.circuits):
        logger.info(f"Circuit {i+1}: {circuit.name}")
        logger.info(f"  Components: {len(circuit.components)}")
        logger.info(f"  Weights: {len(circuit.weight_params)}")
        logger.info(f"  Flow: {' -> '.join(circuit.flow[:5])}{'...' if len(circuit.flow) > 5 else ''}")
        if circuit.validation:
            logger.info(f"  Superadditive: {circuit.validation.superadditive}")
        logger.info(f"  Confidence: {circuit.confidence:.2f}")
        logger.info("")

    # Cleanup
    del model, capture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return circuit_results.to_dict()


__all__ = ["run_probe_analysis"]
