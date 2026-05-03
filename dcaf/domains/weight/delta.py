"""
Projection-level weight delta extraction and RMS normalization.

def:weight-delta:
  ΔW_i^(proj) = W_{i,peak}^(proj) - W_0^(proj)

def:rms-norm:
  ||ΔW||_RMS = ||ΔW||_F / √(m·n)
"""

import math
from typing import Dict

import torch
from torch import Tensor

from dcaf.core.defaults import EPS_RMS


def compute_projection_rms(delta: Tensor) -> float:
    """
    RMS-normalized Frobenius norm of a projection delta.

    ||ΔW_i^(proj)||_RMS = ||ΔW_i^(proj)||_F / √(m·n)

    This normalizes for projection size, making norms comparable
    across projections of different dimensions (e.g., attention Q
    at [256, 2304] vs MLP gate at [9216, 2304]).

    Args:
        delta: 2D weight delta tensor

    Returns:
        RMS-normalized Frobenius norm
    """
    m, n = delta.shape
    frob = torch.norm(delta, p="fro").item()
    return frob / math.sqrt(m * n)


def compute_base_relative_delta(
    delta: Tensor,
    W_base: Tensor,
    eps_rms: float = EPS_RMS,
) -> float:
    """
    Base-weight relative normalization (diagnostic metadata only).

    ||ΔW||_rel = ||ΔW||_F / (||W₀||_F + ε)

    NOT used in any discovery or confidence formulas. Stored in
    ProjectionResult.base_relative_delta for interpretation.

    Args:
        delta: 2D weight delta tensor
        W_base: 2D base weight tensor (from M₀)
        eps_rms: Numerical stability floor

    Returns:
        Relative delta magnitude
    """
    frob_delta = torch.norm(delta, p="fro").item()
    frob_base = torch.norm(W_base, p="fro").item()
    return frob_delta / (frob_base + eps_rms)


def compute_all_projection_rms(
    proj_deltas: Dict[str, Tensor],
) -> Dict[str, float]:
    """
    Compute RMS-normalized norms for all projections.

    Args:
        proj_deltas: {proj_id: delta_tensor} for one signal

    Returns:
        {proj_id: rms_norm}
    """
    return {
        proj_id: compute_projection_rms(delta)
        for proj_id, delta in proj_deltas.items()
    }


__all__ = [
    "compute_projection_rms",
    "compute_base_relative_delta",
    "compute_all_projection_rms",
]
