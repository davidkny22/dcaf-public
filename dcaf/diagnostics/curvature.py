"""
Online curvature tracking during training (§7, Def 7.2).

Measures whether training took a straight or winding path through
activation space. A curvature of 0 means the path was perfectly straight;
higher values indicate more winding trajectories.

    curvature = (cumulative_path_length / direct_distance) - 1

The tracker is designed for online use: initialize once at the start of
training, call update at each evaluation step, and finalize at the end.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class CurvatureMetrics:
    """
    Curvature metrics for a training run.

    Attributes:
        enabled: Whether curvature tracking was active.
        curvature_plus: Mean curvature across T+ signals (None if disabled).
        curvature_minus: Mean curvature across T- signals (None if disabled).
    """

    enabled: bool
    curvature_plus: Optional[float] = None
    curvature_minus: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "curvature_plus": self.curvature_plus,
            "curvature_minus": self.curvature_minus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurvatureMetrics":
        """Load from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            curvature_plus=data.get("curvature_plus"),
            curvature_minus=data.get("curvature_minus"),
        )


@dataclass
class OnlineCurvatureTracker:
    """
    Tracks the curvature of activation trajectories for a single component.

    Stores the initial activations (set once), the previous activations
    (updated at each eval step), and the cumulative path length (running
    sum of segment lengths between consecutive evaluation points).

    Attributes:
        component: Component ID (e.g., "L10H3", "L12_MLP").
        initial_activations: Activations captured at the start of training.
        prev_activations: Most recent activations (updated each eval step).
        cumulative_path_length: Running sum of Euclidean segment lengths.
    """

    component: str
    initial_activations: torch.Tensor
    prev_activations: torch.Tensor
    cumulative_path_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (tensors stored as lists for JSON)."""
        return {
            "component": self.component,
            "initial_activations": self.initial_activations.cpu().tolist(),
            "prev_activations": self.prev_activations.cpu().tolist(),
            "cumulative_path_length": self.cumulative_path_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineCurvatureTracker":
        """Load from dictionary."""
        return cls(
            component=data["component"],
            initial_activations=torch.tensor(data["initial_activations"]),
            prev_activations=torch.tensor(data["prev_activations"]),
            cumulative_path_length=data.get("cumulative_path_length", 0.0),
        )


def _capture_activations(
    model: torch.nn.Module,
    component: str,
    probe_data: Any,
    capture_fn: Callable,
) -> torch.Tensor:
    """
    Capture activations for a component using the provided capture function.

    Args:
        model: The model to capture activations from.
        component: Component ID to capture.
        probe_data: Data to feed through the model for activation capture.
        capture_fn: Callable(model, component, probe_data) -> Tensor.

    Returns:
        Flattened activation tensor.
    """
    activations = capture_fn(model, component, probe_data)
    return activations.flatten().float()


def init_curvature_tracker(
    model: torch.nn.Module,
    component: str,
    probe_data: Any,
    capture_fn: Callable,
) -> OnlineCurvatureTracker:
    """
    Initialize a curvature tracker with the current activation state.

    Should be called once at the start of training before any parameter
    updates have occurred.

    Args:
        model: The model to capture initial activations from.
        component: Component ID (e.g., "L10H3").
        probe_data: Data to feed through the model for activation capture.
        capture_fn: Callable(model, component, probe_data) -> Tensor.

    Returns:
        OnlineCurvatureTracker initialized with the current activations.
    """
    activations = _capture_activations(model, component, probe_data, capture_fn)
    logger.debug(
        "Initialized curvature tracker for %s (activation dim=%d)",
        component, activations.numel(),
    )
    return OnlineCurvatureTracker(
        component=component,
        initial_activations=activations.detach().clone(),
        prev_activations=activations.detach().clone(),
        cumulative_path_length=0.0,
    )


def update_curvature_tracker(
    tracker: OnlineCurvatureTracker,
    model: torch.nn.Module,
    probe_data: Any,
    capture_fn: Callable,
) -> None:
    """
    Update the tracker with the current activation state.

    Computes the Euclidean distance from the previous activations to the
    current activations and adds it to the cumulative path length.

    Should be called at each evaluation checkpoint during training.

    Args:
        tracker: The tracker to update (modified in place).
        model: The model to capture current activations from.
        probe_data: Data to feed through the model for activation capture.
        capture_fn: Callable(model, component, probe_data) -> Tensor.
    """
    current = _capture_activations(model, tracker.component, probe_data, capture_fn)

    # Compute segment length (Euclidean distance from previous to current)
    segment_length = torch.norm(current - tracker.prev_activations).item()
    tracker.cumulative_path_length += segment_length

    # Update rolling previous activations
    tracker.prev_activations = current.detach().clone()

    logger.debug(
        "Curvature update for %s: segment=%.4f, cumulative=%.4f",
        tracker.component, segment_length, tracker.cumulative_path_length,
    )


def finalize_curvature(
    tracker: OnlineCurvatureTracker,
    model: torch.nn.Module,
    probe_data: Any,
    capture_fn: Callable,
) -> float:
    """
    Finalize curvature computation and return the curvature metric.

    Captures the final activations, adds the last segment to the path
    length, then computes:

        curvature = (cumulative_path_length / direct_distance) - 1

    A curvature of 0 means the path was perfectly straight (all movement
    in one direction). Higher values indicate more winding paths.

    Args:
        tracker: The tracker to finalize.
        model: The model to capture final activations from.
        probe_data: Data to feed through the model for activation capture.
        capture_fn: Callable(model, component, probe_data) -> Tensor.

    Returns:
        Curvature value (0 = straight, higher = more winding).
    """
    final = _capture_activations(model, tracker.component, probe_data, capture_fn)

    # Add the final segment
    final_segment = torch.norm(final - tracker.prev_activations).item()
    tracker.cumulative_path_length += final_segment

    # Direct distance: start to finish
    direct_distance = torch.norm(final - tracker.initial_activations).item()

    if direct_distance < 1e-8:
        # No net movement — curvature is undefined; return 0 if path was
        # also zero, otherwise infinity-like large value
        if tracker.cumulative_path_length < 1e-8:
            curvature = 0.0
        else:
            curvature = float("inf")
        logger.warning(
            "Near-zero direct distance for %s (path=%.6f, direct=%.6f) — curvature=%s",
            tracker.component, tracker.cumulative_path_length, direct_distance, curvature,
        )
        return curvature

    curvature = (tracker.cumulative_path_length / direct_distance) - 1.0

    logger.debug(
        "Curvature for %s: path=%.4f, direct=%.4f, curvature=%.4f",
        tracker.component, tracker.cumulative_path_length, direct_distance, curvature,
    )

    return curvature


__all__ = [
    "CurvatureMetrics",
    "OnlineCurvatureTracker",
    "init_curvature_tracker",
    "update_curvature_tracker",
    "finalize_curvature",
]
