"""
Contrastive direction extraction.

Implements def:contrastive-direction, def:direction-emergence, and
def:direction-rotation from sec:geometry-analysis:

  d_i^(k) = (Σ_w + λI)^{-1}(μ+ - μ-) / ‖(Σ_w + λI)^{-1}(μ+ - μ-)‖

  Δ‖d‖^(k) = ‖d_post‖ - ‖d_pre‖

  ρ^(k) = cos(d_pre, d_post)
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from dcaf.core.defaults import EPS_GENERAL, LAMBDA_CONTRASTIVE

DirectionMethod = Literal["whitened_svd", "dim"]


@dataclass
class WhitenedSVDResult:
    """Whitened SVD extraction result for one activation contrast."""

    directions: Tensor
    whitened_directions: Tensor
    singular_values: Tensor
    variance_explained: float
    condition_number: float
    effective_rank: float


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


def _ensure_2d(activations: Tensor) -> Tensor:
    """Flatten non-batch activation dimensions into a feature axis."""
    if activations.dim() < 2:
        raise ValueError("activations must have shape [n_samples, ...]")
    if activations.dim() > 2:
        return activations.reshape(activations.shape[0], -1)
    return activations


def extract_dim_direction(
    A_plus: Tensor,
    A_minus: Tensor,
    eps: float = EPS_GENERAL,
) -> Tensor:
    """Extract a difference-in-means direction."""
    A_plus = _ensure_2d(A_plus.float())
    A_minus = _ensure_2d(A_minus.float())
    direction = A_plus.mean(dim=0) - A_minus.mean(dim=0)
    norm = torch.norm(direction)
    if norm < eps:
        return direction
    return direction / norm


def extract_whitened_lda_direction(
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
    A_plus = _ensure_2d(A_plus.float())
    A_minus = _ensure_2d(A_minus.float())

    # Compute class means
    mu_plus = A_plus.mean(dim=0)
    mu_minus = A_minus.mean(dim=0)

    # Mean difference
    mean_diff = mu_plus - mu_minus

    # Compute pooled covariance
    sigma_w = compute_pooled_covariance(A_plus, A_minus)

    # Regularized inverse
    d = sigma_w.shape[0]
    sigma_reg = sigma_w + lambda_reg * torch.eye(d, device=A_plus.device, dtype=A_plus.dtype)

    # Solve linear system instead of explicit inverse
    direction = torch.linalg.solve(sigma_reg, mean_diff)

    # Normalize
    norm = torch.norm(direction)
    if norm < eps:
        return direction  # Return unnormalized if near-zero

    return direction / norm


def extract_whitened_svd_directions(
    A_plus: Tensor,
    A_minus: Tensor,
    n_directions: int = 1,
    regularization_eps: float = 1e-4,
    min_variance_ratio: float = 0.01,
    eps: float = EPS_GENERAL,
) -> WhitenedSVDResult:
    """Extract directions with benign-covariance-whitened SVD.

    The negative/benign class defines the baseline covariance. Matched
    positive-vs-negative differences are then decomposed in whitened space and
    mapped back into the original activation coordinates.
    """
    A_plus = _ensure_2d(A_plus.float())
    A_minus = _ensure_2d(A_minus.float())

    if A_plus.shape != A_minus.shape:
        raise ValueError(
            "whitened_svd requires matched A_plus/A_minus activations with the same shape"
        )
    if n_directions < 1:
        raise ValueError("n_directions must be >= 1")

    n_samples, hidden_dim = A_minus.shape
    if n_samples < 2:
        raise ValueError("whitened_svd requires at least two matched activation pairs")

    mu_minus = A_minus.mean(dim=0, keepdim=True)
    minus_centered = A_minus - mu_minus
    cov_minus = (minus_centered.T @ minus_centered) / max(n_samples - 1, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_minus)
    eigenvalues = eigenvalues.clamp(min=0)
    max_eig = eigenvalues.max().item()
    if max_eig <= eps:
        raise ValueError("whitened_svd cannot whiten a near-zero baseline covariance")

    positive_eigs = eigenvalues[eigenvalues > max_eig * 1e-10]
    min_eig = positive_eigs.min().item() if positive_eigs.numel() else eps
    condition_number = max_eig / max(min_eig, eps)

    eig_normalized = eigenvalues / eigenvalues.sum().clamp(min=eps)
    eig_nonzero = eig_normalized[eig_normalized > eps]
    effective_rank = torch.exp(-(eig_nonzero * eig_nonzero.log()).sum()).item()

    valid_mask = eigenvalues > max_eig * min_variance_ratio
    if not torch.any(valid_mask):
        raise ValueError("whitened_svd retained no covariance dimensions")

    eigenvalues_valid = eigenvalues[valid_mask]
    eigenvectors_valid = eigenvectors[:, valid_mask]

    inv_sqrt = 1.0 / torch.sqrt(eigenvalues_valid + regularization_eps)
    whiten_proj = eigenvectors_valid * inv_sqrt.unsqueeze(0)

    plus_whitened = (A_plus - mu_minus) @ whiten_proj
    minus_whitened = minus_centered @ whiten_proj
    diff_whitened = plus_whitened - minus_whitened

    k = min(n_directions, diff_whitened.shape[0], diff_whitened.shape[1])
    if k < 1:
        raise ValueError("whitened_svd has no valid singular directions")

    _, singular_values, vh = torch.linalg.svd(diff_whitened, full_matrices=False)
    whitened_dirs = vh[:k]
    kept_singular_values = singular_values[:k]

    unwhiten_proj = eigenvectors_valid * torch.sqrt(
        eigenvalues_valid + regularization_eps
    ).unsqueeze(0)
    directions = whitened_dirs @ unwhiten_proj.T
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=eps)
    whitened_dirs = whitened_dirs / whitened_dirs.norm(dim=-1, keepdim=True).clamp(min=eps)

    # Orient: direction should point from A_minus mean toward A_plus mean
    mean_diff = A_plus.mean(dim=0) - A_minus.mean(dim=0)
    for i in range(directions.shape[0]):
        if (directions[i] @ mean_diff) < 0:
            directions[i] = -directions[i]
            whitened_dirs[i] = -whitened_dirs[i]

    total_var = (singular_values ** 2).sum().item()
    kept_var = (kept_singular_values ** 2).sum().item()

    return WhitenedSVDResult(
        directions=directions,
        whitened_directions=whitened_dirs,
        singular_values=kept_singular_values,
        variance_explained=kept_var / max(total_var, eps),
        condition_number=condition_number,
        effective_rank=effective_rank,
    )


def extract_contrastive_direction(
    A_plus: Tensor,
    A_minus: Tensor,
    eps: float = EPS_GENERAL,
    method: DirectionMethod = "whitened_svd",
    n_directions: int = 1,
    whitening_eps: float = 1e-4,
    min_variance_ratio: float = 0.01,
) -> Tensor:
    """
    Extract a normalized contrastive direction.

    Args:
        A_plus: Positive class activations [n+, d]
        A_minus: Negative class activations [n-, d]
        eps: Numerical stability
        method: "whitened_svd" (default) or "dim"
        n_directions: Number of directions for methods that support subspaces.
            This wrapper returns the first direction.
        whitening_eps: Covariance regularization for whitened SVD.
        min_variance_ratio: Eigenvalue retention floor for whitened SVD.

    Returns:
        Normalized contrastive direction [d]
    """
    if method == "dim":
        return extract_dim_direction(A_plus, A_minus, eps=eps)
    if method == "whitened_svd":
        return extract_whitened_svd_directions(
            A_plus,
            A_minus,
            n_directions=n_directions,
            regularization_eps=whitening_eps,
            min_variance_ratio=min_variance_ratio,
            eps=eps,
        ).directions[0]

    raise ValueError(f"Unknown direction extraction method: {method!r}")


def extract_contrastive_directions_batch(
    activations_by_signal: Dict[str, Tuple[Tensor, Tensor]],
    method: DirectionMethod = "whitened_svd",
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
        signal: extract_contrastive_direction(
            A_plus,
            A_minus,
            method=method,
        )
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
    "WhitenedSVDResult",
    "compute_pooled_covariance",
    "extract_dim_direction",
    "extract_whitened_svd_directions",
    "extract_contrastive_direction",
    "extract_contrastive_directions_batch",
    "compute_direction_emergence",
    "compute_direction_rotation",
    "compute_direction_dynamics",
    "aggregate_directions",
]
