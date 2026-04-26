"""
Activation-based discovery runner (H_A) (§3.9-3.12 Activation Discovery).

Identifies leverage points: parameters where small weight changes
produce large activation effects.

Two-stage process:
  Stage 1: K_flagged = {k : m_agg(k) >= Φ_70th}  (component screening)
  Stage 2: S_A(p) = (m_agg(k) / max_m) * w_param(p)
  H_A = {p : S_A(p) >= Φ_85th AND k ∈ K_flagged}
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, Any, Optional, List

import torch
from torch import Tensor

from dcaf.ablation import ModelStateManager
from dcaf.core.defaults import TAU_ACT, TAU_COMP
from dcaf.discovery.activation import compute_activation_discovery_set
from dcaf.domains.activation import (
    ActivationCapture,
    ProbeSet,
    compute_magnitude_from_snapshots,
)
from dcaf.storage.delta_store import DeltaStore

logger = logging.getLogger(__name__)


def run_activation_discovery(
    run_path: Path,
    model_name: str,
    tau_comp: float = TAU_COMP,
    tau_act: float = TAU_ACT,
    probe_size: int = 50,
    device: str = "cuda",
) -> Tuple[Set[int], Dict[int, float], List[str]]:
    """
    Run activation-based discovery from CLI.

    Steps:
    1. Load model and base checkpoint
    2. Create probe set and capture baseline activations
    3. For each delta: apply, capture, compute magnitudes vs baseline
    4. Build param_to_component mapping
    5. Run two-stage discovery (component screening + parameter filtering)

    Args:
        run_path: Path to DCAF run directory
        model_name: Model name to load
        tau_comp: Component screening percentile (70th - generous)
        tau_act: Parameter filtering percentile (85th - strict)
        probe_size: Number of probe prompts
        device: Device for model

    Returns:
        (H_A set of param indices, {param_index: S_A score})
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load delta store and metadata
    delta_store = DeltaStore(run_path)
    available_deltas = delta_store.list_deltas()

    if not available_deltas:
        logger.error(f"No deltas found in {run_path}")
        return set(), {}

    logger.info(f"Found {len(available_deltas)} deltas: {available_deltas}")

    # Create probe set
    metadata = delta_store.load_metadata()
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

    # Load base model (the original pretrained model IS the base state)
    logger.info(f"\nLoading base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize activation capture
    capture = ActivationCapture(model)

    # Capture baseline activations (base model)
    logger.info("\nCapturing baseline activations...")
    baseline_snapshot = capture.capture(
        probe_set, tokenizer, name="baseline", probe_type="both"
    )

    # Get all component names
    all_components: Set[str] = set()
    all_components.update(baseline_snapshot.attention_patterns.keys())
    all_components.update(baseline_snapshot.mlp_activations.keys())
    logger.info(f"  Components captured: {len(all_components)}")

    # Build param_to_component mapping from model's own parameters
    param_names = [name for name, _ in model.named_parameters()]
    param_to_component: Dict[int, str] = {}
    param_magnitudes: Dict[int, float] = {}

    for idx, pname in enumerate(param_names):
        # Extract component from param name (e.g., "model.layers.5.mlp.gate_proj.weight" -> "L5_MLP_G")
        component = extract_component_from_param_name(pname)
        param_to_component[idx] = component

        # Get param magnitude from base model
        param = dict(model.named_parameters()).get(pname)
        if param is not None:
            param_magnitudes[idx] = torch.abs(param.data).mean().item()
        else:
            param_magnitudes[idx] = 0.0

    # Find primary target-side delta for state manager
    safety_delta_name = next(
        (d for d in available_deltas if "t1_prefopt_target" in d),
        next((d for d in available_deltas if "target" in d),
             next((d for d in available_deltas if "safe" in d), available_deltas[0]))
    )
    safety_delta = delta_store.load_delta(safety_delta_name)

    # Use model's current weights as the base state (freshly loaded = base)
    base_state = {name: param.data.clone() for name, param in model.named_parameters()}
    state_manager = ModelStateManager(
        model, base_state, safety_delta, delta_scale=1.0
    )

    # For each delta, capture activations and compute magnitudes
    magnitudes_by_signal: Dict[str, Dict[str, float]] = {}

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
            probe_set, tokenizer, name=signal_name, probe_type="both"
        )

        # Compute magnitudes (comparing to baseline)
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
        magnitudes_by_signal[signal_name] = magnitudes

        logger.info(f"  Computed magnitudes for {len(magnitudes)} components")

    # Compute H_A using two-stage discovery
    logger.info(f"\nComputing H_A with tau_comp={tau_comp}, tau_act={tau_act}...")

    all_params = set(range(len(param_names)))

    H_A, S_A = compute_activation_discovery_set(
        all_params=all_params,
        param_to_component=param_to_component,
        param_magnitudes=param_magnitudes,
        magnitudes_by_signal=magnitudes_by_signal,
        tau_comp=tau_comp,
        tau_act=tau_act,
    )

    # Log summary
    if S_A:
        nonzero_scores = [s for s in S_A.values() if s > 0]
        if nonzero_scores:
            mean_score = sum(nonzero_scores) / len(nonzero_scores)
            max_score = max(nonzero_scores)
            logger.info(f"Mean S_A (nonzero): {mean_score:.4f}")
            logger.info(f"Max S_A: {max_score:.4f}")
    logger.info(f"H_A size: {len(H_A)}")

    # Cleanup
    del model, capture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return H_A, S_A, param_names


def extract_component_from_param_name(param_name: str) -> str:
    """
    Extract component ID from parameter name.

    Examples:
        "model.layers.5.mlp.gate_proj.weight" -> "L5_MLP_G"
        "model.layers.10.self_attn.q_proj.weight" -> "L10_Q"
        "model.embed_tokens.weight" -> "embed"
        "lm_head.weight" -> "lm_head"

    Args:
        param_name: Full parameter name

    Returns:
        Component identifier
    """
    parts = param_name.split(".")

    # Check for layer pattern
    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                pass

    if layer_idx is not None:
        # Check for MLP vs attention
        if "mlp" in param_name:
            if "gate" in param_name:
                return f"L{layer_idx}_MLP_G"
            elif "up" in param_name:
                return f"L{layer_idx}_MLP_U"
            elif "down" in param_name:
                return f"L{layer_idx}_MLP_D"
            else:
                return f"L{layer_idx}_MLP"
        elif "self_attn" in param_name or "attention" in param_name:
            if "q_proj" in param_name:
                return f"L{layer_idx}_Q"
            elif "k_proj" in param_name:
                return f"L{layer_idx}_K"
            elif "v_proj" in param_name:
                return f"L{layer_idx}_V"
            elif "o_proj" in param_name:
                return f"L{layer_idx}_O"
            else:
                return f"L{layer_idx}_ATTN"
        elif "layernorm" in param_name.lower() or "ln" in param_name.lower():
            if "input" in param_name or "pre" in param_name:
                return f"L{layer_idx}_LN_pre"
            else:
                return f"L{layer_idx}_LN_post"
        else:
            return f"L{layer_idx}_other"

    # Non-layer params
    if "embed" in param_name:
        return "embed"
    elif "lm_head" in param_name:
        return "lm_head"
    elif "norm" in param_name.lower():
        return "final_norm"
    else:
        return "other"


__all__ = ["run_activation_discovery", "extract_component_from_param_name"]
