"""
Contrastive direction extraction.

Implements Def 6.2 (Whitened Contrastive Direction), Def 6.4 (Direction
Emergence), Def 6.5 (Direction Rotation) from §6 (Geometry Analysis):

  d_i^(k) = (Σ_w + λI)^{-1}(μ+ - μ-) / ‖(Σ_w + λI)^{-1}(μ+ - μ-)‖

  Δ‖d‖^(k) = ‖d_post‖ - ‖d_pre‖

  ρ^(k) = cos(d_pre, d_post)
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F

from dcaf.core.defaults import LAMBDA_CONTRASTIVE, EPS_GENERAL


@dataclass
class DirectionDynamics:
    """
    Direction dynamics result.

    Attributes:
        delta_norm: Emergence Δ‖d‖^(k) = ‖d_post‖ - ‖d_pre‖
        rotation: Rotation ρ^(k) = cos(d_pre, d_post)
        d_pre_norm: Pre-training direction norm
        d_post_norm: Post-training direction norm
    """
    delta_norm: float
    rotation: float
    d_pre_norm: float
    d_post_norm: float


def compute_pooled_covariance(
    A_plus: Tensor,
    A_minus: Tensor,
) -> Tensor:
    """
    Compute pooled within-class covariance.

    Σ_w = (Cov(A+) + Cov(A-))/2

    Args:
        A_plus: Positive class activations [n+, d]
        A_minus: Negative class activations [n-, d]

    Returns:
        Pooled covariance [d, d]
    """
    # Center the data
    A_plus_centered = A_plus - A_plus.mean(dim=0, keepdim=True)
    A_minus_centered = A_minus - A_minus.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    n_plus = A_plus.shape[0]
    n_minus = A_minus.shape[0]

    if n_plus > 1:
        cov_plus = (A_plus_centered.T @ A_plus_centered) / (n_plus - 1)
    else:
        cov_plus = torch.zeros(A_plus.shape[1], A_plus.shape[1], device=A_plus.device)

    if n_minus > 1:
        cov_minus = (A_minus_centered.T @ A_minus_centered) / (n_minus - 1)
    else:
        cov_minus = torch.zeros(A_minus.shape[1], A_minus.shape[1], device=A_minus.device)

    # Pool covariances
    return (cov_plus + cov_minus) / 2


def extract_contrastive_direction(
    A_plus: Tensor,
    A_minus: Tensor,
    lambda_reg: float = LAMBDA_CONTRASTIVE,
    eps: float = EPS_GENERAL,
) -> Tensor:
    """
    Extract whitened contrastive direction.

    d_i^(k) = (Σ_w + λI)^{-1}(μ+ - μ-) / ‖(Σ_w + λI)^{-1}(μ+ - μ-)‖

    Args:
        A_plus: Positive class activations [n+, d]
        A_minus: Negative class activations [n-, d]
        lambda_reg: Regularization parameter
        eps: Numerical stability

    Returns:
        Normalized contrastive direction [d]
    """
    # Compute class means
    mu_plus = A_plus.mean(dim=0)
    mu_minus = A_minus.mean(dim=0)

    # Mean difference
    mean_diff = mu_plus - mu_minus

    # Compute pooled covariance
    sigma_w = compute_pooled_covariance(A_plus, A_minus)

    # Regularized inverse
    d = sigma_w.shape[0]
    sigma_reg = sigma_w + lambda_reg * torch.eye(d, device=A_plus.device)

    # Solve linear system instead of explicit inverse
    direction = torch.linalg.solve(sigma_reg, mean_diff)

    # Normalize
    norm = torch.norm(direction)
    if norm < eps:
        return direction  # Return unnormalized if near-zero

    return direction / norm


def extract_contrastive_directions_batch(
    activations_by_signal: Dict[str, Tuple[Tensor, Tensor]],
    lambda_reg: float = LAMBDA_CONTRASTIVE,
) -> Dict[str, Tensor]:
    """
    Extract contrastive directions for multiple signals.

    Args:
        activations_by_signal: {signal_id: (A_plus, A_minus)}
        lambda_reg: Regularization parameter

    Returns:
        {signal_id: direction}
    """
    return {
        signal: extract_contrastive_direction(A_plus, A_minus, lambda_reg)
        for signal, (A_plus, A_minus) in activations_by_signal.items()
    }


def compute_direction_emergence(
    d_pre: Tensor,
    d_post: Tensor,
) -> float:
    """
    Compute direction emergence (norm change).

    Δ‖d‖^(k) = ‖d_post‖ - ‖d_pre‖

    Args:
        d_pre: Pre-training direction
        d_post: Post-training direction

    Returns:
        Emergence value (positive = direction strengthened)
    """
    return torch.norm(d_post).item() - torch.norm(d_pre).item()


def compute_direction_rotation(
    d_pre: Tensor,
    d_post: Tensor,
    eps: float = EPS_GENERAL,
) -> float:
    """
    Compute direction rotation (cosine similarity).

    ρ^(k) = cos(d_pre, d_post)

    Args:
        d_pre: Pre-training direction
        d_post: Post-training direction
        eps: Numerical stability

    Returns:
        Rotation value in [-1, 1]
            ρ ≈ 1: Same direction
            ρ ∈ [0.5, 0.9]: Partial rotation
            ρ < 0.5: Substantial rotation
    """
    norm_pre = torch.norm(d_pre)
    norm_post = torch.norm(d_post)

    if norm_pre < eps or norm_post < eps:
        return 0.0

    return F.cosine_similarity(
        d_pre.unsqueeze(0),
        d_post.unsqueeze(0)
    ).item()


def compute_direction_dynamics(
    d_pre: Tensor,
    d_post: Tensor,
) -> DirectionDynamics:
    """
    Compute full direction dynamics (emergence and rotation).

    Args:
        d_pre: Pre-training direction
        d_post: Post-training direction

    Returns:
        DirectionDynamics with emergence and rotation
    """
    return DirectionDynamics(
        delta_norm=compute_direction_emergence(d_pre, d_post),
        rotation=compute_direction_rotation(d_pre, d_post),
        d_pre_norm=torch.norm(d_pre).item(),
        d_post_norm=torch.norm(d_post).item(),
    )


def aggregate_directions(
    directions: Dict[str, Tensor],
    signal_ids: Optional[list] = None,
) -> Tensor:
    """
    Aggregate contrastive directions across signals.

    Args:
        directions: {signal_id: direction}
        signal_ids: Optional subset of signals to aggregate

    Returns:
        Mean direction (normalized)
    """
    if signal_ids is None:
        signal_ids = list(directions.keys())

    if not signal_ids:
        raise ValueError("No signals to aggregate")

    # Stack and mean
    direction_stack = torch.stack([directions[s] for s in signal_ids])
    mean_direction = direction_stack.mean(dim=0)

    # Normalize
    norm = torch.norm(mean_direction)
    if norm > 0:
        mean_direction = mean_direction / norm

    return mean_direction


__all__ = [
    "DirectionDynamics",
    "compute_pooled_covariance",
    "extract_contrastive_direction",
    "extract_contrastive_directions_batch",
    "compute_direction_emergence",
    "compute_direction_rotation",
    "compute_direction_dynamics",
    "aggregate_directions",
]
