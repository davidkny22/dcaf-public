"""
Nonlinear representation diagnostics.

When the Linear Representation Score (LRS) falls below a threshold
(default 0.4), the behavior may be gated, distributed, or encoded
nonlinearly. This module provides diagnostic tools to characterize
the nonlinear structure of activation spaces using PaCMAP embeddings
and Procrustes alignment.

Diagnostics:
- PaCMAP silhouette on raw activations: Cluster separability in 2D
- PaCMAP silhouette on activation deltas: Change-space separability
- Procrustes structural mirror score: Alignment between T+ and T- embeddings
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging

import numpy as np
import torch
from torch import Tensor

# Lazy imports for optional dependencies (not available in all environments)
_pacmap = None
_silhouette_score = None
_scipy_spatial = None


def _ensure_nonlinear_deps():
    """Import optional dependencies on first use."""
    global _pacmap, _silhouette_score, _scipy_spatial
    if _pacmap is None:
        try:
            import pacmap as _pm
            _pacmap = _pm
        except ImportError:
            raise ImportError(
                "pacmap is required for nonlinear diagnostics. "
                "Install with: pip install pacmap"
            )
    if _silhouette_score is None:
        from sklearn.metrics import silhouette_score as _ss
        _silhouette_score = _ss
    if _scipy_spatial is None:
        import scipy.spatial as _sp
        _scipy_spatial = _sp

logger = logging.getLogger(__name__)


@dataclass
class NonlinearDiagnostics:
    """
    Nonlinear representation diagnostic results.

    Only populated when LRS < threshold, indicating the linear
    representation hypothesis is weak and nonlinear analysis is warranted.

    Attributes:
        triggered: Whether diagnostics were triggered (LRS < threshold)
        lrs: The LRS value that triggered this diagnostic
        pacmap_A_silhouette: Silhouette score on PaCMAP of raw activations
        pacmap_deltaA_silhouette: Silhouette score on PaCMAP of activation deltas
        procrustes_structural_mirror_score: Procrustes alignment between
            T+ and T- embeddings (1 = perfect mirror, 0 = no structure)
    """
    triggered: bool
    lrs: float
    pacmap_A_silhouette: float
    pacmap_deltaA_silhouette: float
    procrustes_structural_mirror_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "triggered": self.triggered,
            "lrs": self.lrs,
            "pacmap_A_silhouette": self.pacmap_A_silhouette,
            "pacmap_deltaA_silhouette": self.pacmap_deltaA_silhouette,
            "procrustes_structural_mirror_score": self.procrustes_structural_mirror_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NonlinearDiagnostics":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with diagnostic fields.

        Returns:
            NonlinearDiagnostics instance.
        """
        return cls(
            triggered=data["triggered"],
            lrs=data["lrs"],
            pacmap_A_silhouette=data["pacmap_A_silhouette"],
            pacmap_deltaA_silhouette=data["pacmap_deltaA_silhouette"],
            procrustes_structural_mirror_score=data["procrustes_structural_mirror_score"],
        )


def compute_pacmap_on_activations(
    component: str,
    activation_snapshots: Dict[str, Tensor],
    signals: list,
) -> float:
    """
    Compute silhouette score on PaCMAP embedding of raw activations.

    Stacks activations from T+ and T- signals, embeds them into 2D
    via PaCMAP, and evaluates cluster separability using silhouette score.

    Args:
        component: Component identifier (e.g., "L10H3") for logging.
        activation_snapshots: Mapping from signal ID to activation tensor.
            Each tensor has shape [n_samples, ...] (will be flattened).
        signals: List of signal objects with .id and .cluster attributes.
            Cluster values: '+' for T+, '-' for T-, '0' for baseline.

    Returns:
        Silhouette score in [-1, 1]. Higher values indicate better
        separation between T+ and T- clusters in the embedded space.
    """
    _ensure_nonlinear_deps()

    activations_list = []
    labels_list = []

    for signal in signals:
        if signal.cluster not in ('+', '-'):
            continue
        if signal.id not in activation_snapshots:
            continue

        act = activation_snapshots[signal.id]
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()

        # Flatten to 2D: [n_samples, features]
        if act.ndim > 2:
            act = act.reshape(act.shape[0], -1)

        label = 1 if signal.cluster == '+' else 0
        activations_list.append(act)
        labels_list.extend([label] * act.shape[0])

    if len(activations_list) < 2:
        logger.warning(
            "Component %s: insufficient signals for PaCMAP on activations "
            "(need T+ and T-), returning 0.0",
            component,
        )
        return 0.0

    X = np.concatenate(activations_list, axis=0)
    labels = np.array(labels_list)

    # Need at least 2 unique labels for silhouette
    if len(np.unique(labels)) < 2:
        logger.warning(
            "Component %s: only one cluster present, returning 0.0",
            component,
        )
        return 0.0

    logger.debug(
        "Component %s: running PaCMAP on activations, shape %s",
        component, X.shape,
    )

    embedding = _pacmap.PaCMAP(n_components=2).fit_transform(X)
    score = _silhouette_score(embedding, labels)

    logger.debug(
        "Component %s: PaCMAP activation silhouette = %.4f",
        component, score,
    )

    return float(score)


def compute_pacmap_on_deltas(
    component: str,
    activation_deltas: Dict[str, Tensor],
    signals: list,
) -> float:
    """
    Compute silhouette score on PaCMAP embedding of activation deltas.

    Same approach as compute_pacmap_on_activations but operates on
    activation deltas (one delta vector per signal, flattened).

    Args:
        component: Component identifier for logging.
        activation_deltas: Mapping from signal ID to delta tensor.
            Each tensor is a single flattened delta vector or
            a batch of delta vectors [n_samples, ...].
        signals: List of signal objects with .id and .cluster attributes.

    Returns:
        Silhouette score in [-1, 1].
    """
    _ensure_nonlinear_deps()

    deltas_list = []
    labels_list = []

    for signal in signals:
        if signal.cluster not in ('+', '-'):
            continue
        if signal.id not in activation_deltas:
            continue

        delta = activation_deltas[signal.id]
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()

        # Ensure 2D
        if delta.ndim == 1:
            delta = delta.reshape(1, -1)
        elif delta.ndim > 2:
            delta = delta.reshape(delta.shape[0], -1)

        label = 1 if signal.cluster == '+' else 0
        deltas_list.append(delta)
        labels_list.extend([label] * delta.shape[0])

    if len(deltas_list) < 2:
        logger.warning(
            "Component %s: insufficient signals for PaCMAP on deltas "
            "(need T+ and T-), returning 0.0",
            component,
        )
        return 0.0

    X = np.concatenate(deltas_list, axis=0)
    labels = np.array(labels_list)

    if len(np.unique(labels)) < 2:
        logger.warning(
            "Component %s: only one cluster present in deltas, returning 0.0",
            component,
        )
        return 0.0

    logger.debug(
        "Component %s: running PaCMAP on deltas, shape %s",
        component, X.shape,
    )

    embedding = _pacmap.PaCMAP(n_components=2).fit_transform(X)
    score = _silhouette_score(embedding, labels)

    logger.debug(
        "Component %s: PaCMAP delta silhouette = %.4f",
        component, score,
    )

    return float(score)


def compute_procrustes_alignment(
    component: str,
    activation_deltas: Dict[str, Tensor],
    signals: list,
) -> float:
    """
    Compute Procrustes structural mirror score between T+ and T- embeddings.

    Runs PaCMAP separately on T+ deltas and T- deltas, then aligns the
    two embeddings via _scipy_spatial.procrustes. The structural mirror
    score measures how well the T+ and T- activation structures mirror
    each other (indicating opposing but structurally similar representations).

    structural_mirror_score = 1 - (disparity / baseline)

    where baseline is the Procrustes disparity of the unaligned embeddings.

    Args:
        component: Component identifier for logging.
        activation_deltas: Mapping from signal ID to delta tensor.
        signals: List of signal objects with .id and .cluster attributes.

    Returns:
        Structural mirror score in [0, 1]. Higher values indicate
        that T+ and T- have structurally mirrored representations.
    """
    _ensure_nonlinear_deps()

    plus_deltas = []
    minus_deltas = []

    for signal in signals:
        if signal.id not in activation_deltas:
            continue

        delta = activation_deltas[signal.id]
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()

        # Flatten to 1D per signal
        if delta.ndim > 1:
            delta = delta.flatten()

        if signal.cluster == '+':
            plus_deltas.append(delta)
        elif signal.cluster == '-':
            minus_deltas.append(delta)

    if len(plus_deltas) < 2 or len(minus_deltas) < 2:
        logger.warning(
            "Component %s: insufficient signals for Procrustes alignment "
            "(need >= 2 T+ and >= 2 T-), returning 0.0",
            component,
        )
        return 0.0

    X_plus = np.array(plus_deltas)
    X_minus = np.array(minus_deltas)

    logger.debug(
        "Component %s: running PaCMAP for Procrustes, T+ shape %s, T- shape %s",
        component, X_plus.shape, X_minus.shape,
    )

    # Embed each cluster separately
    embedding_plus = _pacmap.PaCMAP(n_components=2).fit_transform(X_plus)
    embedding_minus = _pacmap.PaCMAP(n_components=2).fit_transform(X_minus)

    # Align sizes: use the smaller of the two
    n = min(embedding_plus.shape[0], embedding_minus.shape[0])
    embedding_plus = embedding_plus[:n]
    embedding_minus = embedding_minus[:n]

    # Compute baseline disparity (before alignment) using Frobenius norm
    # Standardize both matrices to zero mean, unit scale for fair comparison
    plus_centered = embedding_plus - embedding_plus.mean(axis=0)
    minus_centered = embedding_minus - embedding_minus.mean(axis=0)
    plus_scale = np.sqrt(np.sum(plus_centered ** 2))
    minus_scale = np.sqrt(np.sum(minus_centered ** 2))

    if plus_scale < 1e-10 or minus_scale < 1e-10:
        logger.warning(
            "Component %s: degenerate embeddings, returning 0.0",
            component,
        )
        return 0.0

    baseline = np.sum(
        (plus_centered / plus_scale - minus_centered / minus_scale) ** 2
    )

    if baseline < 1e-10:
        # Already perfectly aligned
        return 1.0

    # Procrustes alignment
    _, _, disparity = _scipy_spatial.procrustes(embedding_plus, embedding_minus)

    # Structural mirror score: higher = more mirrored
    score = 1.0 - (disparity / baseline)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    logger.debug(
        "Component %s: Procrustes mirror score = %.4f (disparity=%.4f, baseline=%.4f)",
        component, score, disparity, baseline,
    )

    return float(score)


def compute_nonlinear_diagnostics(
    component: str,
    lrs: float,
    activation_snapshots: Dict[str, Tensor],
    activation_deltas: Dict[str, Tensor],
    signals: list,
    lrs_threshold: float = 0.4,
) -> Optional[NonlinearDiagnostics]:
    """
    Compute nonlinear representation diagnostics for a component.

    Only runs when LRS < threshold, indicating the linear representation
    hypothesis is weak. When triggered, computes:
    1. PaCMAP silhouette on raw activations
    2. PaCMAP silhouette on activation deltas
    3. Procrustes structural mirror score

    Args:
        component: Component identifier (e.g., "L10H3").
        lrs: Linear Representation Score for this component.
        activation_snapshots: Mapping from signal ID to activation tensor.
        activation_deltas: Mapping from signal ID to delta tensor.
        signals: List of signal objects with .id and .cluster attributes.
        lrs_threshold: LRS threshold below which diagnostics are triggered.

    Returns:
        NonlinearDiagnostics if LRS < threshold, None otherwise.
    """
    if lrs >= lrs_threshold:
        logger.debug(
            "Component %s: LRS=%.4f >= threshold=%.4f, skipping nonlinear diagnostics",
            component, lrs, lrs_threshold,
        )
        return None

    logger.info(
        "Component %s: LRS=%.4f < threshold=%.4f, running nonlinear diagnostics",
        component, lrs, lrs_threshold,
    )

    pacmap_A = compute_pacmap_on_activations(component, activation_snapshots, signals)
    pacmap_deltaA = compute_pacmap_on_deltas(component, activation_deltas, signals)
    procrustes = compute_procrustes_alignment(component, activation_deltas, signals)

    result = NonlinearDiagnostics(
        triggered=True,
        lrs=lrs,
        pacmap_A_silhouette=pacmap_A,
        pacmap_deltaA_silhouette=pacmap_deltaA,
        procrustes_structural_mirror_score=procrustes,
    )

    logger.info(
        "Component %s: nonlinear diagnostics complete — "
        "A_sil=%.4f, dA_sil=%.4f, procrustes=%.4f",
        component, pacmap_A, pacmap_deltaA, procrustes,
    )

    return result


__all__ = [
    "NonlinearDiagnostics",
    "compute_pacmap_on_activations",
    "compute_pacmap_on_deltas",
    "compute_procrustes_alignment",
    "compute_nonlinear_diagnostics",
]
