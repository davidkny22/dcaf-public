"""
Activation magnitude computation.

def:activation-delta-and-magnitude:
m_i^(k,π) = (1/n_π) · Σⱼ ‖ΔA_i^(k,π)[j]‖₂
"""

from typing import Dict

import torch
from torch import Tensor

from dcaf.core.defaults import EPS_GENERAL


def compute_tensor_delta(
    post_activations: Tensor,
    pre_activations: Tensor,
) -> Tensor:
    """
    Compute activation delta tensor from training.

    ΔA_i^(k,π) = A_i^(k,π) - A_0^(k,π)

    Args:
        post_activations: Activations after training [n_π, d_k]
        pre_activations: Activations before training (base model) [n_π, d_k]

    Returns:
        Delta tensor [n_π, d_k]
    """
    return post_activations - pre_activations


def compute_magnitude(
    delta: Tensor,
    eps: float = EPS_GENERAL,
) -> float:
    """
    Compute activation magnitude for a component.

    m_i^(k,π) = (1/n_π) · Σⱼ ‖ΔA[j]‖₂

    Args:
        delta: Activation delta [n_π, d_k] or [n_π, ...]
        eps: Numerical stability

    Returns:
        Scalar magnitude value
    """
    if delta.numel() == 0:
        return 0.0

    # Flatten to [n_examples, features] if needed
    if delta.dim() == 1:
        return delta.abs().mean().item()

    # Reshape to [n_examples, -1] for L2 norm computation
    n_examples = delta.shape[0]
    flat = delta.reshape(n_examples, -1)

    # L2 norm per example, then mean
    norms = torch.norm(flat, p=2, dim=1)  # [n_examples]
    return norms.mean().item()


def compute_magnitude_batch(
    deltas: Dict[str, Tensor],
) -> Dict[str, float]:
    """
    Compute magnitude for multiple components.

    Args:
        deltas: {component_id: delta_tensor}

    Returns:
        {component_id: magnitude}
    """
    return {
        component: compute_magnitude(delta)
        for component, delta in deltas.items()
    }


def compute_magnitude_from_snapshots(
    post_snapshot: Dict[str, Tensor],
    pre_snapshot: Dict[str, Tensor],
) -> Dict[str, float]:
    """
    Compute magnitudes from pre/post activation snapshots.

    Args:
        post_snapshot: {component_id: activations} after training
        pre_snapshot: {component_id: activations} before training

    Returns:
        {component_id: magnitude}
    """
    magnitudes = {}

    for component in post_snapshot.keys():
        if component not in pre_snapshot:
            continue

        delta = compute_tensor_delta(
            post_snapshot[component],
            pre_snapshot[component],
        )
        magnitudes[component] = compute_magnitude(delta)

    return magnitudes


__all__ = [
    "compute_tensor_delta",
    "compute_magnitude",
    "compute_magnitude_batch",
    "compute_magnitude_from_snapshots",
]
