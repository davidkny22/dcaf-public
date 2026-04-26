"""
Named Checkpoint Manager for DCAF training runs (Def 1.10).

Provides save/restore functionality for model checkpoints with named references.
Checkpoints are stored on CPU to minimize GPU memory usage during multi-phase
DCAF training.

Standard checkpoint names (spec-aligned):
- "base"          - W0, the untrained base model state
- "checkpoint_t1" - After prefopt target (SimPO safe) training
- "checkpoint_t2" - After SFT target training
- "checkpoint_t3" - After cumulative target training
- "checkpoint_t4" - After anti target training
- "checkpoint_t5" - After negated target training
- "checkpoint_t6" - After prefopt opposite (SimPO unsafe) training
- "checkpoint_t7" - After SFT opposite training
- "checkpoint_t8" - After cumulative opposite training
- "checkpoint_t9" - After anti opposite training
- "checkpoint_t10" - After negated opposite training
- "checkpoint_t11" - After language baseline training
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import logging

import torch
import numpy as np
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class NamedCheckpoint:
    """A named model checkpoint with metadata."""

    name: str
    weights: Dict[str, torch.Tensor]  # param_name -> weight (CPU)
    timestamp: datetime
    training_phase: Optional[str] = None  # What training produced this
    parent_checkpoint: Optional[str] = None  # Which checkpoint it was derived from

    @property
    def param_count(self) -> int:
        """Number of parameters in this checkpoint."""
        return len(self.weights)

    @property
    def total_elements(self) -> int:
        """Total number of weight elements."""
        return sum(w.numel() for w in self.weights.values())


class CheckpointManager:
    """
    Manages named checkpoints for DCAF variants.

    Provides save/restore/delta operations on model checkpoints.
    All checkpoints are stored on CPU to minimize GPU memory usage.
    """

    # Standard checkpoint names used across variants (spec-aligned)
    STANDARD_NAMES = {
        "base": "W0 - untrained base model",
        "checkpoint_t1": "After prefopt target (SimPO safe)",
        "checkpoint_t2": "After SFT target",
        "checkpoint_t3": "After cumulative target",
        "checkpoint_t4": "After anti target",
        "checkpoint_t5": "After negated target",
        "checkpoint_t6": "After prefopt opposite (SimPO unsafe)",
        "checkpoint_t7": "After SFT opposite",
        "checkpoint_t8": "After cumulative opposite",
        "checkpoint_t9": "After anti opposite",
        "checkpoint_t10": "After negated opposite",
        "checkpoint_t11": "After language baseline",
    }

    def __init__(self, model: PreTrainedModel, device: str = "cuda"):
        """
        Initialize checkpoint manager.

        Args:
            model: The model to manage checkpoints for
            device: Device the model is on (for restore operations)
        """
        self.model = model
        self.device = device
        self._checkpoints: Dict[str, NamedCheckpoint] = {}
        self._deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        self._current_checkpoint_name: Optional[str] = None

    def save_checkpoint(
        self,
        name: str,
        training_phase: Optional[str] = None
    ) -> NamedCheckpoint:
        """
        Save current model state as named checkpoint (CPU storage).

        Args:
            name: Name for this checkpoint
            training_phase: Optional description of what training produced this

        Returns:
            The created NamedCheckpoint
        """
        weights = {}
        for param_name, param in self.model.named_parameters():
            weights[param_name] = param.detach().cpu().clone()

        checkpoint = NamedCheckpoint(
            name=name,
            weights=weights,
            timestamp=datetime.now(),
            training_phase=training_phase,
            parent_checkpoint=self._current_checkpoint_name,
        )

        self._checkpoints[name] = checkpoint
        self._current_checkpoint_name = name

        logger.info(f"  Saved checkpoint '{name}' ({checkpoint.param_count} params)")
        return checkpoint

    def restore_checkpoint(self, name: str) -> int:
        """
        Restore model to named checkpoint.

        Args:
            name: Name of checkpoint to restore

        Returns:
            Count of parameters restored

        Raises:
            KeyError: If checkpoint name not found
        """
        if name not in self._checkpoints:
            available = list(self._checkpoints.keys())
            raise KeyError(f"Checkpoint '{name}' not found. Available: {available}")

        checkpoint = self._checkpoints[name]
        restored = 0

        with torch.no_grad():
            for param_name, param in self.model.named_parameters():
                if param_name in checkpoint.weights:
                    param.copy_(checkpoint.weights[param_name].to(self.device))
                    restored += 1

        self._current_checkpoint_name = name
        logger.info(f"  Restored checkpoint '{name}' ({restored} params)")
        return restored

    def compute_delta(
        self,
        from_checkpoint: str,
        to_checkpoint: str,
        delta_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        Compute delta = to_checkpoint - from_checkpoint.

        Args:
            from_checkpoint: Name of the base checkpoint
            to_checkpoint: Name of the target checkpoint
            delta_name: Name to store this delta under

        Returns:
            Dictionary mapping param names to delta tensors
        """
        if from_checkpoint not in self._checkpoints:
            raise KeyError(f"From checkpoint '{from_checkpoint}' not found")
        if to_checkpoint not in self._checkpoints:
            raise KeyError(f"To checkpoint '{to_checkpoint}' not found")

        from_ckpt = self._checkpoints[from_checkpoint]
        to_ckpt = self._checkpoints[to_checkpoint]

        deltas = {}
        for param_name in from_ckpt.weights.keys():
            if param_name in to_ckpt.weights:
                deltas[param_name] = (
                    to_ckpt.weights[param_name] - from_ckpt.weights[param_name]
                )

        self._deltas[delta_name] = deltas
        logger.info(f"  Computed delta '{delta_name}' ({len(deltas)} params)")
        return deltas

    def get_delta(self, delta_name: str) -> Dict[str, torch.Tensor]:
        """Get previously computed delta by name."""
        if delta_name not in self._deltas:
            raise KeyError(f"Delta '{delta_name}' not found. Available: {list(self._deltas.keys())}")
        return self._deltas[delta_name]

    def get_checkpoint(self, name: str) -> NamedCheckpoint:
        """Get checkpoint by name."""
        if name not in self._checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found")
        return self._checkpoints[name]

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint names."""
        return list(self._checkpoints.keys())

    def list_deltas(self) -> List[str]:
        """List all available delta names."""
        return list(self._deltas.keys())

    def has_checkpoint(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        return name in self._checkpoints

    def has_delta(self, name: str) -> bool:
        """Check if a delta exists."""
        return name in self._deltas

    def compute_delta_magnitude(
        self,
        delta_name: str,
        param_name: str
    ) -> float:
        """Compute L2 norm of a specific parameter's delta."""
        delta = self.get_delta(delta_name)
        if param_name not in delta:
            return 0.0
        return torch.norm(delta[param_name]).item()

    def compute_threshold(
        self,
        delta_name: str,
        percentile: float = 85.0
    ) -> float:
        """
        Compute significance threshold for a delta using percentile.

        Args:
            delta_name: Name of the delta
            percentile: Percentile for threshold (default 85th)

        Returns:
            Threshold value
        """
        delta = self.get_delta(delta_name)
        magnitudes = [torch.norm(d).item() for d in delta.values()]
        return float(np.percentile(magnitudes, percentile))

    def clear(self) -> None:
        """Clear all checkpoints and deltas."""
        self._checkpoints.clear()
        self._deltas.clear()
        self._current_checkpoint_name = None
        logger.info("  Cleared all checkpoints and deltas")

    @property
    def current_checkpoint(self) -> Optional[str]:
        """Name of the most recently saved/restored checkpoint."""
        return self._current_checkpoint_name


__all__ = [
    "NamedCheckpoint",
    "CheckpointManager",
]
