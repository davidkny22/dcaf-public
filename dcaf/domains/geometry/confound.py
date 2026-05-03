"""
Confound direction and independence.

def:confound-direction; def:confound-independence:
d_c^(k) = whitened_direction(A_c+, A_c-)
ρ_c^(k) = 1 - (1/|T+|) · Σ_{i∈T+} |cos(d_i^(k), d_c^(k))|
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from dcaf.core.defaults import EPS_GENERAL, LAMBDA_CONTRASTIVE

from .directions import extract_contrastive_direction


@dataclass
class ConfoundAnalysis:
    """
    Confound analysis result.

    Attributes:
        confound_direction: The extracted confound direction d_c^(k)
        independence: Confound independence ρ_c^(k)
        per_signal_overlap: {signal: |cos(d_i, d_c)|} for each T+ signal
    """
    confound_direction: Tensor
    independence: float
    per_signal_overlap: Dict[str, float]


def extract_confound_direction(
    A_confound_plus: Tensor,
    A_confound_minus: Tensor,
    lambda_reg: float = LAMBDA_CONTRASTIVE,
) -> Tensor:
    """
    Extract confound direction from confound activations.

    d_c^(k) = whitened_direction(A_c+, A_c-)

    Confound examples for safety behavior:
    - A_c+: Generic refusal (out-of-domain, malformed prompts, capability limits)
    - A_c-: Generic compliance on neutral requests

    Args:
        A_confound_plus: Confound-positive activations [n+, d]
        A_confound_minus: Confound-negative activations [n-, d]
        lambda_reg: Regularization parameter

    Returns:
        Normalized confound direction [d]
    """
    return extract_contrastive_direction(
        A_confound_plus,
        A_confound_minus,
        whitening_eps=lambda_reg,
    )


def compute_direction_overlap(
    d_behavioral: Tensor,
    d_confound: Tensor,
    eps: float = EPS_GENERAL,
) -> float:
    """
    Compute overlap between behavioral and confound directions.

    |cos(d_i, d_c)|

    Args:
        d_behavioral: Behavioral direction
        d_confound: Confound direction
        eps: Numerical stability

    Returns:
        Absolute cosine similarity in [0, 1]
    """
    norm_b = torch.norm(d_behavioral)
    norm_c = torch.norm(d_confound)

    if norm_b < eps or norm_c < eps:
        return 0.0

    cos_sim = F.cosine_similarity(
        d_behavioral.unsqueeze(0),
        d_confound.unsqueeze(0)
    ).item()

    return abs(cos_sim)


def compute_confound_independence(
    directions: Dict[str, Tensor],
    d_confound: Tensor,
    T_plus_signals: List[str],
) -> float:
    """
    Compute confound independence for a component.

    ρ_c^(k) = 1 - (1/|T+|) · Σ_{i∈T+} |cos(d_i^(k), d_c^(k))|

    Interpretation:
    - ρ_c > 0.7: Clean — direction captures behavior-specific features
    - ρ_c ∈ [0.3, 0.7]: Mixed signal
    - ρ_c < 0.3: Contaminated — dominated by confound

    Args:
        directions: {signal_id: direction} for all signals
        d_confound: Confound direction
        T_plus_signals: Signal IDs in T+

    Returns:
        Independence value in [0, 1]
    """
    if not T_plus_signals:
        return 1.0

    overlaps = []
    for signal in T_plus_signals:
        if signal in directions:
            overlap = compute_direction_overlap(directions[signal], d_confound)
            overlaps.append(overlap)

    if not overlaps:
        return 1.0

    mean_overlap = sum(overlaps) / len(overlaps)
    return 1.0 - mean_overlap


def compute_confound_analysis(
    directions: Dict[str, Tensor],
    A_confound_plus: Tensor,
    A_confound_minus: Tensor,
    T_plus_signals: List[str],
    lambda_reg: float = LAMBDA_CONTRASTIVE,
) -> ConfoundAnalysis:
    """
    Complete confound analysis for a component.

    Args:
        directions: {signal_id: direction}
        A_confound_plus: Confound-positive activations
        A_confound_minus: Confound-negative activations
        T_plus_signals: Target cluster signal IDs
        lambda_reg: Regularization parameter

    Returns:
        ConfoundAnalysis with direction, independence, and per-signal overlaps
    """
    # Extract confound direction
    d_confound = extract_confound_direction(
        A_confound_plus,
        A_confound_minus,
        lambda_reg,
    )

    # Compute per-signal overlaps
    per_signal_overlap = {}
    for signal in T_plus_signals:
        if signal in directions:
            per_signal_overlap[signal] = compute_direction_overlap(
                directions[signal], d_confound
            )

    # Compute independence
    independence = compute_confound_independence(
        directions, d_confound, T_plus_signals
    )

    return ConfoundAnalysis(
        confound_direction=d_confound,
        independence=independence,
        per_signal_overlap=per_signal_overlap,
    )


def is_contaminated(
    independence: float,
    threshold: float = 0.3,
) -> bool:
    """
    Check if component is contaminated by confound.

    Args:
        independence: ρ_c value
        threshold: Contamination threshold

    Returns:
        True if contaminated (independence below threshold)
    """
    return independence < threshold


def is_clean(
    independence: float,
    threshold: float = 0.7,
) -> bool:
    """
    Check if component is clean from confound.

    Args:
        independence: ρ_c value
        threshold: Clean threshold

    Returns:
        True if clean (independence above threshold)
    """
    return independence > threshold


__all__ = [
    "ConfoundAnalysis",
    "extract_confound_direction",
    "compute_direction_overlap",
    "compute_confound_independence",
    "compute_confound_analysis",
    "is_contaminated",
    "is_clean",
]
