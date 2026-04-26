"""
Model state management for ablation testing (§11, Def 11.1).

Implements the two ablation techniques from Def 11.1:
- Baseline restoration: replace component weights with baseline model (M_0) values
- Mean ablation: subtract the safety delta to restore pre-training weights

Provides ModelStateManager for centralized reset/ablate patterns used by
all ablation strategies throughout §11 phases.
"""

import torch
from contextlib import contextmanager
from typing import Dict, List, Generator, Optional, Union
from pathlib import Path
from transformers import PreTrainedModel


class ModelStateManager:
    """
    Centralized model state management for ablation testing.

    Handles:
    - Loading base checkpoint and safety delta
    - Resetting model to base state (no safety)
    - Applying safety delta
    - Ablating specific parameters
    - Temporary ablation with auto-restore

    Usage:
        manager = ModelStateManager(model, base_checkpoint, safety_delta)
        manager.reset_to_safety()  # Start with safety
        manager.ablate_params(["model.layers.10.mlp.down_proj.weight"])
        # ... test model ...
        manager.reset_to_safety()  # Restore
    """

    def __init__(
        self,
        model: PreTrainedModel,
        base_checkpoint: Dict[str, torch.Tensor],
        safety_delta: Dict[str, torch.Tensor],
        delta_scale: float = 2.0,
        device: str = "cuda",
    ):
        """
        Initialize the state manager.

        Args:
            model: The model to manage (should already be on device)
            base_checkpoint: Base model weights (loaded from base checkpoint)
            safety_delta: Safety training delta (delta for target training run)
            delta_scale: Scale factor for delta (typically 2.0)
            device: Device to move tensors to
        """
        self.model = model
        self.base_checkpoint = base_checkpoint
        self.safety_delta = safety_delta
        self.delta_scale = delta_scale
        self.device = device

        # Track current state
        self._current_state = "unknown"

    @classmethod
    def from_paths(
        cls,
        model: PreTrainedModel,
        base_path: Union[str, Path],
        delta_path: Union[str, Path],
        delta_scale: float = 2.0,
        device: str = "cuda",
    ) -> "ModelStateManager":
        """
        Create a ModelStateManager from checkpoint file paths.

        Args:
            model: The model to manage
            base_path: Path to base checkpoint file
            delta_path: Path to delta checkpoint file
            delta_scale: Scale factor for delta
            device: Device for the model

        Returns:
            Configured ModelStateManager
        """
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
        safety_delta = torch.load(delta_path, map_location="cpu", weights_only=True)
        return cls(model, base_checkpoint, safety_delta, delta_scale, device)

    def reset_to_base(self) -> None:
        """
        Reset model to raw base state (no safety delta applied).

        This is the "unsafe" state where the model has not been safety-trained.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.base_checkpoint:
                    param.copy_(self.base_checkpoint[name].to(self.device))
        self._current_state = "base"

    def reset_to_safety(self) -> None:
        """
        Reset model to safety-trained state (base + delta * scale).

        This applies the safety delta on top of the base weights.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.base_checkpoint:
                    param.copy_(self.base_checkpoint[name].to(self.device))
                if name in self.safety_delta:
                    param.add_(
                        self.safety_delta[name].to(self.device) * self.delta_scale
                    )
        self._current_state = "safety"

    def ablate_params(self, params: List[str]) -> None:
        """
        Remove safety delta from specific parameters.

        This subtracts the delta from the specified parameters, effectively
        removing the safety training for those weights only.

        Args:
            params: List of parameter names to ablate
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params and name in self.safety_delta:
                    param.sub_(
                        self.safety_delta[name].to(self.device) * self.delta_scale
                    )
        self._current_state = f"ablated:{len(params)}"

    def restore_params(self, params: List[str]) -> None:
        """
        Restore safety delta to specific parameters.

        This adds back the delta to parameters that were previously ablated.

        Args:
            params: List of parameter names to restore
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params and name in self.safety_delta:
                    param.add_(
                        self.safety_delta[name].to(self.device) * self.delta_scale
                    )
        self._current_state = "safety"

    @contextmanager
    def temporary_ablation(self, params: List[str]) -> Generator[None, None, None]:
        """
        Context manager for temporary ablation with auto-restore.

        Usage:
            manager.reset_to_safety()
            with manager.temporary_ablation(["model.layers.10.mlp.down_proj.weight"]):
                # Model has param ablated here
                responses = generate(model, prompts)
            # Model is automatically restored to safety state

        Args:
            params: List of parameter names to temporarily ablate

        Yields:
            None (model is modified in place)
        """
        self.ablate_params(params)
        try:
            yield
        finally:
            self.restore_params(params)

    def get_delta_params(self) -> List[str]:
        """Get list of all parameters that have deltas."""
        return list(self.safety_delta.keys())

    def get_delta_magnitude(self, param: str) -> Optional[float]:
        """Get the mean absolute delta for a parameter."""
        if param not in self.safety_delta:
            return None
        return float(self.safety_delta[param].abs().mean())

    @property
    def current_state(self) -> str:
        """Current model state: 'base', 'safety', 'ablated:N', or 'unknown'."""
        return self._current_state

    def __repr__(self) -> str:
        return (
            f"ModelStateManager("
            f"state={self._current_state}, "
            f"scale={self.delta_scale}, "
            f"delta_params={len(self.safety_delta)})"
        )

__all__ = ["ModelStateManager"]
