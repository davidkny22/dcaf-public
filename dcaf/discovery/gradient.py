"""
Gradient-based parameter discovery H_G (§3, Def 3.13-3.14).

Identifies parameters with high behavioral gradients — mathematical predictions
of which parameters would affect behavior if modified.

Behavioral gradient (Def 3.13):
  g(proj) = sum_{i in T+ union T-} eff(i) * ||dO_i/dW_proj||_F

Gradient discovery set (Def 3.14):
  H_G = {proj : g(proj) >= Phi_{tau_grad}}  (tau_grad = 85)

Note: H_G (gradient discovery) is COMPLETELY DIFFERENT from C_G (geometry confidence):
- H_G: Computes gradients of signal objectives (SFT loss, DPO margin, etc.)
- C_G: Computes representation geometry (cluster coherence, opposition, etc.)

Objective functions by signal type (Def 3.13 Behavioral Objective):
- SFT: O = E[-log p(y|x)]
- PrefOpt (DPO/SimPO): O = -E[M(y_w, y_l|x)]
- Anti/Negated: O = +E[M(y_w, y_l|x)] (gradient ascent on opposite behavior)
"""

from typing import Dict, Set, Tuple, Optional, List, Callable, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from dcaf.core.defaults import TAU_GRAD


def _sequence_logprobs(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Compute length-normalized log-probabilities for a sequence."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
    return (token_logps * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)


@dataclass
class SignalObjective:
    """
    Defines how to compute objective/gradient for a training signal.

    Attributes:
        signal_id: Signal identifier
        signal_type: 'SFT', 'PrefOpt', 'Anti', 'Negated', 'DomainNative'
        cluster: '+', '-', or '0'
        compute_fn: Function that computes the objective value
    """
    signal_id: str
    signal_type: str
    cluster: str
    compute_fn: Optional[Callable[[nn.Module, Any], Tensor]] = None


def compute_signal_gradient(
    model: nn.Module,
    objective_fn: Callable[[nn.Module, Any], Tensor],
    data_batch: Any,
) -> Dict[str, Tensor]:
    """
    Compute gradient of objective w.r.t. all parameters.

    Args:
        model: The model to compute gradients for
        objective_fn: Function that takes (model, data) and returns scalar loss
        data_batch: Input data for the objective

    Returns:
        {param_name: gradient_tensor}
    """
    model.zero_grad()

    # Compute objective
    loss = objective_fn(model, data_batch)

    # Backward pass
    loss.backward()

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach()

    model.zero_grad()
    return gradients


def compute_behavioral_gradient_score(
    param_name: str,
    signal_gradients: Dict[str, Dict[str, Tensor]],
    effectiveness: Dict[str, float],
    behavioral_signals: Set[str],
) -> float:
    """
    Compute behavioral gradient score for a parameter.

    g^(p) = Σ_{i∈T⁺∪T⁻} eff(i) · |∂O_i/∂p|

    Args:
        param_name: Parameter name
        signal_gradients: {signal_id: {param_name: gradient}}
        effectiveness: {signal_id: eff_value}
        behavioral_signals: Set of T+ and T- signal IDs

    Returns:
        Behavioral gradient score
    """
    total = 0.0

    for signal_id in behavioral_signals:
        if signal_id not in signal_gradients:
            continue

        grads = signal_gradients[signal_id]
        if param_name not in grads:
            continue

        eff = effectiveness.get(signal_id, 1.0)
        grad_magnitude = torch.abs(grads[param_name]).mean().item()

        total += eff * grad_magnitude

    return total


def compute_gradient_discovery_scores(
    model: nn.Module,
    signal_objectives: List[SignalObjective],
    data_batches: Dict[str, Any],
    effectiveness: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute gradient discovery scores S_G for all parameters.

    g^(p) = Σ_{i∈T⁺∪T⁻} eff(i) · |∂O_i/∂p|

    Args:
        model: The model
        signal_objectives: List of signal objective definitions
        data_batches: {signal_id: data_batch} for each signal
        effectiveness: {signal_id: eff_value}

    Returns:
        {param_name: S_G score}
    """
    # Filter to behavioral signals (T+ and T-)
    behavioral = [s for s in signal_objectives if s.cluster in ['+', '-']]
    behavioral_ids = {s.signal_id for s in behavioral}

    # Compute gradients for each signal
    signal_gradients: Dict[str, Dict[str, Tensor]] = {}

    for signal in behavioral:
        if signal.compute_fn is None:
            continue
        if signal.signal_id not in data_batches:
            continue

        grads = compute_signal_gradient(
            model, signal.compute_fn, data_batches[signal.signal_id]
        )
        signal_gradients[signal.signal_id] = grads

    # Compute S_G for each parameter
    S_G: Dict[str, float] = {}
    param_names = [name for name, _ in model.named_parameters()]

    for param_name in param_names:
        score = compute_behavioral_gradient_score(
            param_name, signal_gradients, effectiveness, behavioral_ids
        )
        S_G[param_name] = score

    return S_G


def compute_gradient_discovery_set(
    model: nn.Module,
    signal_objectives: List[SignalObjective],
    data_batches: Dict[str, Any],
    effectiveness: Dict[str, float],
    tau_grad: float = TAU_GRAD,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Compute gradient discovery set H_G.

    H_G = {p : g^(p) >= Φ_τgrad}

    Args:
        model: The model
        signal_objectives: List of signal objective definitions
        data_batches: {signal_id: data_batch}
        effectiveness: {signal_id: eff_value}
        tau_grad: Discovery threshold percentile

    Returns:
        (H_G set of param names, {param_name: S_G score})
    """
    S_G = compute_gradient_discovery_scores(
        model, signal_objectives, data_batches, effectiveness
    )

    if not S_G:
        return set(), {}

    # Threshold at tau_grad percentile
    scores = list(S_G.values())
    threshold = np.percentile(scores, tau_grad)

    H_G = {name for name, score in S_G.items() if score >= threshold}

    return H_G, S_G


# =============================================================================
# Convenience functions for common objective types
# =============================================================================

def create_sft_objective(
    loss_fn: Optional[Callable] = None,
) -> Callable[[nn.Module, Any], Tensor]:
    """
    Create SFT objective function.

    O = E[-log p(y|x)]
    """
    def objective(model: nn.Module, batch: Any) -> Tensor:
        if loss_fn is not None:
            return loss_fn(model, batch)
        # Default: assume batch is (input_ids, labels)
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        return outputs.loss

    return objective


def create_preference_objective(
    margin_fn: Optional[Callable] = None,
    negate: bool = False,
) -> Callable[[nn.Module, Any], Tensor]:
    """
    Create preference optimization objective function.

    PrefOpt: O = -E[M(y_w, y_l|x)]
    Anti/Negated: O = +E[M(y_w, y_l|x)] (set negate=True)
    """
    def objective(model: nn.Module, batch: Any) -> Tensor:
        if margin_fn is not None:
            margin = margin_fn(model, batch)
        else:
            chosen_ids = batch["chosen_input_ids"].to(next(model.parameters()).device)
            chosen_mask = batch["chosen_attention_mask"].to(next(model.parameters()).device)
            rejected_ids = batch["rejected_input_ids"].to(next(model.parameters()).device)
            rejected_mask = batch["rejected_attention_mask"].to(next(model.parameters()).device)

            chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
            rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

            chosen_logps = _sequence_logprobs(chosen_logits, chosen_ids, chosen_mask)
            rejected_logps = _sequence_logprobs(rejected_logits, rejected_ids, rejected_mask)
            margin = (chosen_logps - rejected_logps).mean()

        return margin if negate else -margin

    return objective


__all__ = [
    "SignalObjective",
    "compute_signal_gradient",
    "compute_behavioral_gradient_score",
    "compute_gradient_discovery_scores",
    "compute_gradient_discovery_set",
    "create_sft_objective",
    "create_preference_objective",
]
