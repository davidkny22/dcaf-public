"""
Activation delta alignment diagnostics (sec:activation-delta-alignment; def:delta-alignment).

Measures whether training signals within each cluster (T+, T-) move
activations in the same direction. High intra-cluster alignment with
cross-cluster opposition indicates clean, well-separated training signals.

Expected values:
    align_plus  > 0.7  (T+ signals move activations in the same direction)
    align_minus > 0.7  (T- signals move activations in the same direction)
    opposition  < -0.5 (T+ and T- deltas oppose each other)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ActivationDeltaAlignment:
    """
    Alignment metrics for activation deltas within and across signal clusters.

    Attributes:
        align_plus: Mean pairwise cosine similarity among T+ signal activation deltas.
        align_minus: Mean pairwise cosine similarity among T- signal activation deltas.
        opposition: Mean cosine between T+ and T- deltas (should be negative if they oppose).
    """

    align_plus: float
    align_minus: float
    opposition: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "align_plus": self.align_plus,
            "align_minus": self.align_minus,
            "opposition": self.opposition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationDeltaAlignment":
        """Load from dictionary."""
        return cls(
            align_plus=data.get("align_plus", 0.0),
            align_minus=data.get("align_minus", 0.0),
            opposition=data.get("opposition", 0.0),
        )


def _mean_pairwise_cosine(vectors: List[torch.Tensor]) -> float:
    """
    Compute mean pairwise cosine similarity among a list of flattened vectors.

    Args:
        vectors: List of 1-D tensors (already flattened).

    Returns:
        Mean pairwise cosine similarity, or 0.0 if fewer than 2 vectors.
    """
    n = len(vectors)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(
                vectors[i].unsqueeze(0),
                vectors[j].unsqueeze(0),
            ).item()
            total += sim
            count += 1

    return total / count if count > 0 else 0.0


def _mean_cross_cosine(
    group_a: List[torch.Tensor],
    group_b: List[torch.Tensor],
) -> float:
    """
    Compute mean cosine similarity between all pairs across two groups.

    Args:
        group_a: First group of 1-D tensors.
        group_b: Second group of 1-D tensors.

    Returns:
        Mean cross-group cosine similarity, or 0.0 if either group is empty.
    """
    if not group_a or not group_b:
        return 0.0

    total = 0.0
    count = 0
    for va in group_a:
        for vb in group_b:
            sim = F.cosine_similarity(
                va.unsqueeze(0),
                vb.unsqueeze(0),
            ).item()
            total += sim
            count += 1

    return total / count if count > 0 else 0.0


def compute_activation_delta_alignment(
    component: str,
    signals: List,
    activation_deltas: Dict,
) -> Optional[ActivationDeltaAlignment]:
    """
    Compute activation delta alignment for a single component.

    Measures intra-cluster coherence (T+ and T- separately) and
    cross-cluster opposition to assess training signal quality.

    Args:
        component: Component ID (e.g., "L10H3", "L12_MLP").
        signals: List of signal objects with a ``.cluster`` attribute
            ('+', '-', or '0') and an ``.id`` or similar identifier.
        activation_deltas: Mapping of signal_id -> component_id -> activation
            delta tensor. Each delta is the change in activations caused by
            that training signal at the given component.

    Returns:
        ActivationDeltaAlignment with align_plus, align_minus, opposition,
        or None if there are insufficient deltas.
    """
    plus_deltas: List[torch.Tensor] = []
    minus_deltas: List[torch.Tensor] = []

    for signal in signals:
        cluster = getattr(signal, "cluster", None)
        if cluster not in ("+", "-"):
            continue

        # Support both attribute-based and string-based signal IDs
        signal_id = getattr(signal, "id", None) or getattr(signal, "signal_id", None) or str(signal)

        component_deltas = activation_deltas.get(signal_id, {})
        delta = component_deltas.get(component)
        if delta is None:
            continue

        # Flatten to 1-D for cosine similarity
        flat = delta.flatten().float()
        if flat.numel() == 0:
            continue

        if cluster == "+":
            plus_deltas.append(flat)
        elif cluster == "-":
            minus_deltas.append(flat)

    # Truncate ALL deltas (plus + minus) to a shared minimum length
    # so cross-cosine between plus[i] and minus[j] works
    all_deltas = plus_deltas + minus_deltas
    if all_deltas:
        min_len = min(v.numel() for v in all_deltas)
        plus_deltas = [v[:min_len] for v in plus_deltas]
        minus_deltas = [v[:min_len] for v in minus_deltas]

    if not plus_deltas and not minus_deltas:
        logger.debug("No activation deltas found for component %s", component)
        return None

    align_plus = _mean_pairwise_cosine(plus_deltas)
    align_minus = _mean_pairwise_cosine(minus_deltas)
    opposition = _mean_cross_cosine(plus_deltas, minus_deltas)

    logger.debug(
        "Delta alignment for %s: align+=%.3f, align-=%.3f, opp=%.3f",
        component, align_plus, align_minus, opposition,
    )

    return ActivationDeltaAlignment(
        align_plus=align_plus,
        align_minus=align_minus,
        opposition=opposition,
    )


__all__ = [
    "ActivationDeltaAlignment",
    "compute_activation_delta_alignment",
]
