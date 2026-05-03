"""
Gradient-based discovery runner (H_G) (sec:gradient-discovery).

Identifies parameters with high behavioral gradients — mathematical
predictions of what would affect behavior if modified.

  g(p) = Σ_{i∈T⁺∪T⁻} eff(i) · |∂O_i/∂p|
  H_G = {p : g(p) >= Phi_taugrad}

Objective functions by signal type:
  SFT:           O = E[-log p(y|x)]
  PrefOpt:       O = -E[M(y_w, y_l|x)]
  Anti/Negated:  O = +E[M(y_w, y_l|x)]
"""

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dcaf.arch.transformer import should_exclude_param
from dcaf.core.defaults import TAU_GRAD
from dcaf.core.signals import CANONICAL_SIGNALS
from dcaf.discovery.gradient import (
    SignalObjective,
    compute_gradient_discovery_set,
)
from dcaf.domains.weight import (
    compute_effectiveness_from_training_metrics,
    create_uniform_effectiveness,
)
from dcaf.storage.delta_store import DeltaStore

logger = logging.getLogger(__name__)


def run_gradient_discovery(
    run_path: Path,
    model_name: str,
    tau_grad: float = TAU_GRAD,
    device: str = "cuda",
    max_samples: int = 10,
) -> Tuple[Set[int], Dict[int, float], List[str]]:
    """
    Run gradient-based discovery from CLI.

    Steps:
    1. Load model and deltas
    2. For each behavioral signal (T+ ∪ T-):
       - Build objective function based on signal type
       - Load or generate sample data for the signal
       - Compute gradients ∂O_i/∂p for each parameter
       - Weight by effectiveness
    3. Sum across signals: g(p) = Σ eff(i)·|∂O_i/∂p|
    4. Threshold at tau_grad percentile
    5. Convert param names to int indices (matching weight/activation)

    Note: This is computationally expensive. For large models, consider
    using --gradient only when needed.

    Args:
        run_path: Path to DCAF run directory
        model_name: Model name to load
        tau_grad: Gradient discovery percentile threshold
        device: Device for model
        max_samples: Max samples per signal for gradient computation

    Returns:
        (H_G set of param indices, {param_index: S_G score}, param_names list)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load delta store
    delta_store = DeltaStore(run_path)
    available_deltas = delta_store.list_deltas()

    if not available_deltas:
        logger.error(f"No deltas found in {run_path}")
        return set(), {}, []

    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")

    # Load base model and restore the exact checkpoint saved during training.
    logger.info(f"\nLoading base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_checkpoint = delta_store.load_checkpoint("base")
    except FileNotFoundError:
        logger.warning("No saved base checkpoint found; using freshly loaded model as base")
    else:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in base_checkpoint:
                    param.copy_(base_checkpoint[name].to(device))

    # Identify behavioral signals (T+ and T-)
    behavioral_signals = []
    for delta_name in available_deltas:
        cluster = get_signal_cluster(delta_name)
        if cluster in ['+', '-']:
            signal_type = infer_signal_type(delta_name)
            behavioral_signals.append({
                'id': delta_name,
                'cluster': cluster,
                'type': signal_type,
            })

    logger.info("\nBehavioral signals for gradient computation:")
    for sig in behavioral_signals:
        logger.info(f"  - {sig['id']} (T{sig['cluster']}, {sig['type']})")

    if not behavioral_signals:
        logger.warning("No behavioral signals found for gradient discovery")
        return set(), {}, []

    # Create effectiveness scores from training metrics when available.
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
    else:
        logger.warning("No training metrics found; using uniform effectiveness")
        effectiveness = create_uniform_effectiveness(available_deltas)

    # Build signal objectives
    signal_objectives = []
    data_batches = {}

    for sig in behavioral_signals:
        # Create objective function based on signal type
        objective_fn = create_objective_function(sig['type'], model, tokenizer, device)

        signal_objectives.append(SignalObjective(
            signal_id=sig['id'],
            signal_type=sig['type'],
            cluster=sig['cluster'],
            compute_fn=objective_fn,
        ))

        # Generate sample data for this signal
        data_batches[sig['id']] = create_sample_batch(
            signal_type=sig['type'],
            tokenizer=tokenizer,
            device=device,
            max_samples=max_samples,
        )

    # Compute gradient discovery set
    logger.info("\nComputing behavioral gradients...")
    logger.info("This may take a while for large models...")

    from dcaf.core.topology import build_model_topology
    topo = build_model_topology(model)
    logger.info(f"  Topology: {len(topo.projections)} projections — projection-level gradients")

    H_G_str, S_G_str = compute_gradient_discovery_set(
        model=model,
        signal_objectives=signal_objectives,
        data_batches=data_batches,
        effectiveness=effectiveness,
        tau_grad=tau_grad,
        topology=topo,
    )

    # Build name_to_index mapping from delta store (same ordering as weight/activation)
    all_delta_params = set()
    for delta_name in available_deltas:
        all_delta_params.update(delta_store.load_delta(delta_name).keys())
    param_names_list = sorted(all_delta_params)
    name_to_index = {name: idx for idx, name in enumerate(param_names_list)}

    # Convert str keys to int indices
    H_G: Set[int] = set()
    for name in H_G_str:
        if should_exclude_param(name):
            continue
        if name in name_to_index:
            H_G.add(name_to_index[name])
        else:
            logger.warning(f"Gradient param '{name}' not found in delta store param ordering")

    S_G: Dict[int, float] = {}
    for name, score in S_G_str.items():
        if should_exclude_param(name):
            continue
        if name in name_to_index:
            S_G[name_to_index[name]] = score

    # Log summary
    if S_G:
        nonzero_scores = [s for s in S_G.values() if s > 0]
        if nonzero_scores:
            mean_score = sum(nonzero_scores) / len(nonzero_scores)
            max_score = max(nonzero_scores)
            logger.info(f"Mean S_G (nonzero): {mean_score:.6f}")
            logger.info(f"Max S_G: {max_score:.6f}")
    logger.info(f"H_G size: {len(H_G)}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return H_G, S_G, param_names_list


def get_signal_cluster(delta_name: str) -> str:
    """
    Determine cluster (T+, T-, T0) for a delta name.

    Checks canonical signal IDs first (delta_t1_prefopt_target, etc.),
    then falls back to name-based heuristics for any non-canonical names.

    Args:
        delta_name: Name of the delta (e.g. "delta_t1_prefopt_target")

    Returns:
        '+', '-', or '0'
    """
    # Match against canonical signal ids (e.g. delta_t1, delta_t1_prefopt_target)
    for sig in CANONICAL_SIGNALS:
        if delta_name == f"delta_{sig.id}" or delta_name.startswith(f"delta_{sig.id}_"):
            return sig.cluster
    # Heuristic fallback
    n = delta_name.lower()
    if "language" in n or "neutral" in n or "baseline" in n or n == "delta_t11_baseline":
        return "0"
    if "target" in n or ("safe" in n and "anti" not in n and "negated" not in n):
        return "+"
    if "opposite" in n or ("adv" in n and "anti" not in n):
        return "-"
    # Default: treat as behavioral T+
    return "+"


def infer_signal_type(delta_name: str) -> str:
    """
    Infer signal type from delta name.

    Args:
        delta_name: Name of the delta

    Returns:
        'SFT', 'PrefOpt', 'Anti', 'Negated', or 'Unknown'
    """
    name_lower = delta_name.lower()

    if 'sft' in name_lower:
        return 'SFT'
    elif 'simpo' in name_lower or 'dpo' in name_lower or 'pref' in name_lower:
        if 'anti' in name_lower:
            return 'Anti'
        elif 'negated' in name_lower:
            return 'Negated'
        else:
            return 'PrefOpt'
    elif 'anti' in name_lower:
        return 'Anti'
    elif 'negated' in name_lower:
        return 'Negated'
    else:
        return 'PrefOpt'  # Default to preference optimization


def create_objective_function(
    signal_type: str,
    model: nn.Module,
    tokenizer: Any,
    device: str,
) -> Callable[[nn.Module, Any], Tensor]:
    """
    Create objective function for gradient computation.

    Args:
        signal_type: Type of signal ('SFT', 'PrefOpt', 'Anti', 'Negated')
        model: The model
        tokenizer: The tokenizer
        device: Device

    Returns:
        Function that computes scalar objective from model and batch
    """
    def sft_objective(model: nn.Module, batch: Dict) -> Tensor:
        """SFT objective: O = E[-log p(y|x)]"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return outputs.loss

    def pref_objective(model: nn.Module, batch: Dict) -> Tensor:
        """
        PrefOpt objective: O = -E[M(y_w, y_l|x)]

        For simplicity, we approximate margin as difference in log probs.
        """
        # Compute log probs for chosen
        chosen_outputs = model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask'],
        )
        chosen_logits = chosen_outputs.logits
        chosen_log_probs = compute_log_probs(chosen_logits, batch['chosen_input_ids'])

        # Compute log probs for rejected
        rejected_outputs = model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask'],
        )
        rejected_logits = rejected_outputs.logits
        rejected_log_probs = compute_log_probs(rejected_logits, batch['rejected_input_ids'])

        # Margin = chosen - rejected (higher is better)
        margin = chosen_log_probs.mean() - rejected_log_probs.mean()

        # PrefOpt minimizes -margin
        return -margin

    def anti_objective(model: nn.Module, batch: Dict) -> Tensor:
        """Anti objective: O = +E[M(y_w, y_l|x)] (gradient ascent)"""
        return -pref_objective(model, batch)

    if signal_type == 'SFT':
        return sft_objective
    elif signal_type in ['Anti', 'Negated']:
        return anti_objective
    else:  # PrefOpt
        return pref_objective


def compute_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Compute log probabilities of labels given logits.

    Args:
        logits: Model logits [batch, seq, vocab]
        labels: Label token IDs [batch, seq]

    Returns:
        Log probabilities [batch]
    """
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probs
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probs for labels
    label_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum over sequence, mean over batch
    return label_log_probs.sum(dim=-1)


def create_sample_batch(
    signal_type: str,
    tokenizer: Any,
    device: str,
    max_samples: int = 10,
) -> Dict[str, Tensor]:
    """
    Create sample batch for gradient computation.

    Args:
        signal_type: Type of signal
        tokenizer: The tokenizer
        device: Device
        max_samples: Maximum number of samples

    Returns:
        Batch dictionary with tensors
    """
    from dcaf.data.test_banks import get_benign_test_prompts, get_refusal_test_prompts

    if signal_type == 'SFT':
        # For SFT, use simple prompts with completions
        prompts = get_benign_test_prompts(format="instruction")[:max_samples]
        texts = [p + " Sure, I can help with that." for p in prompts]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': encoded['input_ids'].clone(),
        }

    else:  # PrefOpt, Anti, Negated
        # For preference, use chosen/rejected pairs
        harmful_prompts = get_refusal_test_prompts(format="question")[:max_samples]

        chosen_texts = [p + " I can't help with that request." for p in harmful_prompts]
        rejected_texts = [p + " Sure, here's how to do that." for p in harmful_prompts]

        chosen_encoded = tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        rejected_encoded = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        return {
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask'],
        }


__all__ = [
    "run_gradient_discovery",
    "get_signal_cluster",
    "infer_signal_type",
]
