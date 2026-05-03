"""
Predictivity computation.

def:direction-predictivity; def:predictivity-gain:
pred(d) = AUC({a_j · d}, {y_j})
Δ_pred^(k) = (1/|T+|) · Σ_{i∈T+} (pred_post - pred_pre)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from torch import Tensor


@dataclass
class PredictivityResult:
    """
    Predictivity result for a direction.

    Attributes:
        auc: AUC-ROC score in [0, 1]
        projections: Projection values for each sample
        labels: Binary labels for each sample
        n_samples: Number of samples used
    """
    auc: float
    projections: Tensor
    labels: Tensor
    n_samples: int


def compute_auc(
    projections: Tensor,
    labels: Tensor,
) -> float:
    """
    Compute AUC-ROC from projections and labels.

    Uses a simple O(n log n) algorithm without sklearn dependency.

    Args:
        projections: Projection scores [n]
        labels: Binary labels [n] (0 or 1)

    Returns:
        AUC score in [0, 1]
    """
    # Convert to numpy for easier manipulation
    proj_np = projections.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Sort by projection descending
    sorted_indices = np.argsort(-proj_np)
    sorted_labels = labels_np[sorted_indices]

    # Count positives and negatives
    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return chance level

    # Compute AUC using Mann-Whitney U statistic with tie handling
    sorted_proj = proj_np[sorted_indices]
    u_stat = 0.0
    cum_pos = 0
    i = 0
    n = len(sorted_proj)
    while i < n:
        # Find all tied scores
        j = i
        while j < n and sorted_proj[j] == sorted_proj[i]:
            j += 1
        # Count positives and negatives in this tie group
        tie_pos = int(np.sum(sorted_labels[i:j] == 1))
        tie_neg = int(np.sum(sorted_labels[i:j] == 0))
        # Negatives in tie group: each gets credit for all positives ranked
        # strictly higher, plus half credit for positives in the same tie group
        u_stat += tie_neg * (cum_pos + 0.5 * tie_pos)
        cum_pos += tie_pos
        i = j

    return u_stat / (n_pos * n_neg)


def compute_predictivity(
    direction: Tensor,
    activations: Tensor,
    labels: Tensor,
) -> PredictivityResult:
    """
    Compute predictivity of a direction.

    pred(d) = AUC({a_j · d}, {y_j})

    Args:
        direction: Contrastive direction [d]
        activations: Sample activations [n, d]
        labels: Binary labels [n] (0 = negative, 1 = positive)

    Returns:
        PredictivityResult with AUC and details
    """
    # Project activations onto direction
    projections = activations @ direction

    # Compute AUC
    auc = compute_auc(projections, labels)

    return PredictivityResult(
        auc=auc,
        projections=projections,
        labels=labels,
        n_samples=len(labels),
    )


def compute_predictivity_gain(
    d_pre: Tensor,
    d_post: Tensor,
    activations: Tensor,
    labels: Tensor,
) -> float:
    """
    Compute predictivity gain from training.

    Δ_pred = pred_post(d_post) - pred_pre(d_pre)

    Args:
        d_pre: Pre-training direction
        d_post: Post-training direction
        activations: Sample activations
        labels: Binary labels

    Returns:
        Predictivity gain (positive = improved)
    """
    pred_pre = compute_predictivity(d_pre, activations, labels)
    pred_post = compute_predictivity(d_post, activations, labels)

    return pred_post.auc - pred_pre.auc


def compute_predictivity_gain_batch(
    directions_pre: Dict[str, Tensor],
    directions_post: Dict[str, Tensor],
    activations: Tensor,
    labels: Tensor,
    T_plus_signals: List[str],
) -> float:
    """
    Compute average predictivity gain across T+ signals.

    Δ_pred^(k) = (1/|T+|) · Σ_{i∈T+} (pred_post(d_i) - pred_pre(d_i))

    Args:
        directions_pre: {signal_id: pre-direction}
        directions_post: {signal_id: post-direction}
        activations: Sample activations
        labels: Binary labels
        T_plus_signals: Target cluster signal IDs

    Returns:
        Mean predictivity gain
    """
    gains = []

    for signal in T_plus_signals:
        if signal in directions_pre and signal in directions_post:
            gain = compute_predictivity_gain(
                directions_pre[signal],
                directions_post[signal],
                activations,
                labels,
            )
            gains.append(gain)

    if not gains:
        return 0.0

    return sum(gains) / len(gains)


def normalize_predictivity_gain(
    delta_pred: float,
) -> float:
    """
    Normalize predictivity gain to [0, 1].

    x_6 = (1 + Δ_pred) / 2

    Maps Δ_pred ∈ [-1, 1] to [0, 1].

    Args:
        delta_pred: Raw predictivity gain

    Returns:
        Normalized value in [0, 1]
    """
    return (1.0 + delta_pred) / 2.0


def compute_predictivity_at_threshold(
    direction: Tensor,
    activations: Tensor,
    labels: Tensor,
    threshold: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute accuracy and balanced accuracy at threshold.

    Args:
        direction: Contrastive direction
        activations: Sample activations
        labels: Binary labels
        threshold: Decision threshold

    Returns:
        (accuracy, balanced_accuracy)
    """
    projections = activations @ direction
    predictions = (projections > threshold).float()

    # Accuracy
    correct = (predictions == labels).float()
    accuracy = correct.mean().item()

    # Balanced accuracy
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() > 0:
        tpr = correct[pos_mask].mean().item()
    else:
        tpr = 0.0

    if neg_mask.sum() > 0:
        tnr = correct[neg_mask].mean().item()
    else:
        tnr = 0.0

    balanced_acc = (tpr + tnr) / 2.0

    return accuracy, balanced_acc


__all__ = [
    "PredictivityResult",
    "compute_auc",
    "compute_predictivity",
    "compute_predictivity_gain",
    "compute_predictivity_gain_batch",
    "normalize_predictivity_gain",
    "compute_predictivity_at_threshold",
]
