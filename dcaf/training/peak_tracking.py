"""
Peak Checkpoint Tracking for DCAF Training.

Implements stability-confirmed peak detection during preference training.
Instead of using the final checkpoint or a simple best-metric checkpoint,
this module tracks peaks that are confirmed stable over subsequent evaluations.

Algorithm:
    1. Always track the raw best (highest metric value).
    2. A new peak becomes a "candidate" when it exceeds all previous metrics.
    3. A candidate is "confirmed" if the next K evaluations stay within
       delta (relative tolerance) of the candidate metric.
    4. If subsequent metrics drop below tolerance, the candidate is discarded.
    5. At finalization: use the confirmed peak if available, otherwise fall
       back to the raw best.

The tracker MAXIMIZES the metric. Callers must invert if needed (e.g.,
pass -loss for SFT training so that lower loss = higher metric).

Memory: O(2) checkpoints at most (one confirmed + one candidate), plus
the raw best weights. In the common case where the confirmed peak IS the
raw best, this collapses to O(2).

Usage:
    state = PeakTrackingState()

    for step in training_steps:
        if step % eval_interval == 0:
            metric = -loss  # invert for maximization
            weights = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            state = update_peak_tracking(state, step, metric, weights, config)

    result = finalize_peak_tracking(state)
    # result.peak_weights contains the best stable checkpoint
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import logging

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PeakTrackingConfig:
    """Configuration for peak checkpoint tracking.

    Attributes:
        peak_eval_interval: Evaluate peak tracking every N training steps.
        peak_confirmation_window: K -- number of subsequent evaluations that
            must remain within tolerance to confirm a candidate peak.
        peak_stability_tolerance: Delta -- maximum relative drop from the
            candidate metric that still counts as "stable". A value of 0.05
            means the metric may drop up to 5% relative to the candidate.
    """

    peak_eval_interval: int = 50
    peak_confirmation_window: int = 3
    peak_stability_tolerance: float = 0.05


# ---------------------------------------------------------------------------
# Result dataclass -- returned after training completes
# ---------------------------------------------------------------------------

@dataclass
class CheckpointHistory:
    """Result of training with peak tracking.

    Contains the selected peak checkpoint weights, the metric trajectory,
    and metadata about which checkpoint was chosen and why.

    Attributes:
        peak_step: Training step at which the selected peak occurred.
        peak_metric: Metric value at the selected peak.
        peak_weights: Model weights (param_name -> CPU tensor) at the peak.
        is_confirmed: True if the peak was stability-confirmed; False if
            we fell back to the raw best.
        confirmation_count: Number of stable evaluations that confirmed the
            peak (0 if not confirmed).
        metric_history: List of (step, metric) tuples for the full run.
        raw_best_step: Step of the raw (unconfirmed) best metric.
        raw_best_metric: Value of the raw best metric.
        candidate_discard_count: Number of candidate peaks that were
            discarded due to instability during the run.
    """

    peak_step: int
    peak_metric: float
    peak_weights: Dict[str, torch.Tensor]
    is_confirmed: bool
    confirmation_count: int
    metric_history: List[tuple] = field(default_factory=list)
    raw_best_step: int = 0
    raw_best_metric: float = float("-inf")
    candidate_discard_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (excludes weights)."""
        return {
            "peak_step": self.peak_step,
            "peak_metric": self.peak_metric,
            "is_confirmed": self.is_confirmed,
            "confirmation_count": self.confirmation_count,
            "metric_history": self.metric_history,
            "raw_best_step": self.raw_best_step,
            "raw_best_metric": self.raw_best_metric,
            "candidate_discard_count": self.candidate_discard_count,
        }


# ---------------------------------------------------------------------------
# Internal state machine
# ---------------------------------------------------------------------------

@dataclass
class PeakTrackingState:
    """Internal state machine for stability-confirmed peak detection.

    This is mutated step-by-step via :func:`update_peak_tracking` and
    converted into a :class:`CheckpointHistory` via
    :func:`finalize_peak_tracking`.

    Attributes:
        metric_history: Accumulated (step, metric) pairs.
        raw_best_metric: Highest metric observed so far.
        raw_best_step: Step at which the raw best occurred.
        raw_best_weights: Weights snapshot at the raw best.
        candidate_metric: Metric of the current candidate peak (or None).
        candidate_step: Step of the current candidate peak (or None).
        candidate_weights: Weights snapshot of the candidate (or None).
        candidate_confirmations: Number of subsequent evaluations that
            stayed within tolerance of the candidate.
        confirmed_metric: Metric of the most recently confirmed peak (or None).
        confirmed_step: Step of the confirmed peak (or None).
        confirmed_weights: Weights of the confirmed peak (or None).
        confirmed_count: Number of confirmations the confirmed peak received.
        candidate_discard_count: How many candidates were discarded.
    """

    metric_history: List[tuple] = field(default_factory=list)

    # Raw best tracking (unconditional)
    raw_best_metric: float = float("-inf")
    raw_best_step: int = 0
    raw_best_weights: Optional[Dict[str, torch.Tensor]] = None

    # Candidate peak (awaiting confirmation)
    candidate_metric: Optional[float] = None
    candidate_step: Optional[int] = None
    candidate_weights: Optional[Dict[str, torch.Tensor]] = None
    candidate_confirmations: int = 0

    # Confirmed peak (stability verified)
    confirmed_metric: Optional[float] = None
    confirmed_step: Optional[int] = None
    confirmed_weights: Optional[Dict[str, torch.Tensor]] = None
    confirmed_count: int = 0

    # Diagnostics
    candidate_discard_count: int = 0


# ---------------------------------------------------------------------------
# State machine update
# ---------------------------------------------------------------------------

def update_peak_tracking(
    state: PeakTrackingState,
    step: int,
    metric: float,
    weights: Dict[str, torch.Tensor],
    config: PeakTrackingConfig,
) -> PeakTrackingState:
    """Advance the peak tracking state machine with a new evaluation.

    This should be called at each evaluation point (every
    ``config.peak_eval_interval`` steps). The function updates the state
    in-place and also returns it for convenience.

    Args:
        state: Current tracking state.
        step: Current training step number.
        metric: Evaluation metric at this step. Higher is better; callers
            must invert loss-based metrics (e.g. pass ``-loss``).
        weights: Snapshot of model weights at this step. Must be on CPU.
            The caller is responsible for ``.detach().cpu().clone()``.
        config: Peak tracking configuration.

    Returns:
        The updated ``PeakTrackingState`` (same object, mutated in place).
    """
    state.metric_history.append((step, metric))

    # --- 1. Update raw best ---
    if metric > state.raw_best_metric:
        state.raw_best_metric = metric
        state.raw_best_step = step
        state.raw_best_weights = weights
        logger.debug(
            "Peak tracking: new raw best %.6f at step %d", metric, step
        )

    # --- 2. Candidate management ---
    if state.candidate_metric is not None:
        # We have an active candidate -- check stability
        tolerance_floor = state.candidate_metric * (
            1.0 - config.peak_stability_tolerance
        )

        if metric >= tolerance_floor:
            # Metric is within tolerance -- count as confirmation
            state.candidate_confirmations += 1
            logger.debug(
                "Peak tracking: candidate at step %d confirmed %d/%d "
                "(metric=%.6f, floor=%.6f)",
                state.candidate_step,
                state.candidate_confirmations,
                config.peak_confirmation_window,
                metric,
                tolerance_floor,
            )

            if state.candidate_confirmations >= config.peak_confirmation_window:
                # Candidate is now confirmed
                logger.info(
                    "Peak tracking: CONFIRMED peak at step %d "
                    "(metric=%.6f, %d confirmations)",
                    state.candidate_step,
                    state.candidate_metric,
                    state.candidate_confirmations,
                )
                state.confirmed_metric = state.candidate_metric
                state.confirmed_step = state.candidate_step
                state.confirmed_weights = state.candidate_weights
                state.confirmed_count = state.candidate_confirmations

                # Clear candidate slot
                state.candidate_metric = None
                state.candidate_step = None
                state.candidate_weights = None
                state.candidate_confirmations = 0
        else:
            # Metric dropped below tolerance -- discard candidate
            logger.info(
                "Peak tracking: DISCARDED candidate at step %d "
                "(metric dropped to %.6f, floor was %.6f)",
                state.candidate_step,
                metric,
                tolerance_floor,
            )
            state.candidate_discard_count += 1
            state.candidate_metric = None
            state.candidate_step = None
            state.candidate_weights = None
            state.candidate_confirmations = 0

    # --- 3. Check if current metric creates a new candidate ---
    # A new candidate must exceed both the current candidate (if any) and
    # any previously confirmed peak.
    threshold = max(
        state.candidate_metric if state.candidate_metric is not None else float("-inf"),
        state.confirmed_metric if state.confirmed_metric is not None else float("-inf"),
    )

    if metric > threshold and metric >= state.raw_best_metric:
        # New candidate peak
        logger.debug(
            "Peak tracking: new CANDIDATE at step %d (metric=%.6f)",
            step,
            metric,
        )
        state.candidate_metric = metric
        state.candidate_step = step
        state.candidate_weights = weights
        state.candidate_confirmations = 0

    return state


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------

def finalize_peak_tracking(state: PeakTrackingState) -> CheckpointHistory:
    """Convert the tracking state into a final CheckpointHistory.

    Selection logic:
        - If a confirmed peak exists, use it.
        - Otherwise fall back to the raw best checkpoint.

    Args:
        state: The accumulated peak tracking state from training.

    Returns:
        A :class:`CheckpointHistory` with the selected peak weights.

    Raises:
        ValueError: If no evaluations were recorded (empty state).
    """
    if not state.metric_history:
        raise ValueError(
            "Cannot finalize peak tracking: no evaluations were recorded. "
            "Ensure update_peak_tracking() is called at least once."
        )

    if state.confirmed_weights is not None:
        # Use confirmed peak
        logger.info(
            "Peak tracking finalized: using CONFIRMED peak at step %d "
            "(metric=%.6f, %d confirmations)",
            state.confirmed_step,
            state.confirmed_metric,
            state.confirmed_count,
        )
        return CheckpointHistory(
            peak_step=state.confirmed_step,
            peak_metric=state.confirmed_metric,
            peak_weights=state.confirmed_weights,
            is_confirmed=True,
            confirmation_count=state.confirmed_count,
            metric_history=list(state.metric_history),
            raw_best_step=state.raw_best_step,
            raw_best_metric=state.raw_best_metric,
            candidate_discard_count=state.candidate_discard_count,
        )
    else:
        # Fall back to raw best
        if state.raw_best_weights is None:
            raise ValueError(
                "Cannot finalize peak tracking: raw best weights are None. "
                "This should not happen if update_peak_tracking() was called."
            )

        logger.warning(
            "Peak tracking finalized: no confirmed peak found. "
            "Falling back to RAW BEST at step %d (metric=%.6f). "
            "%d candidate(s) were discarded.",
            state.raw_best_step,
            state.raw_best_metric,
            state.candidate_discard_count,
        )
        return CheckpointHistory(
            peak_step=state.raw_best_step,
            peak_metric=state.raw_best_metric,
            peak_weights=state.raw_best_weights,
            is_confirmed=False,
            confirmation_count=0,
            metric_history=list(state.metric_history),
            raw_best_step=state.raw_best_step,
            raw_best_metric=state.raw_best_metric,
            candidate_discard_count=state.candidate_discard_count,
        )


# ---------------------------------------------------------------------------
# HuggingFace TrainerCallback integration
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers import TrainingArguments, PreTrainedModel
    _CALLBACK_BASE = TrainerCallback
except ImportError:
    _CALLBACK_BASE = object
    logger.debug("transformers.TrainerCallback not available; "
                 "PeakTrackingCallback will not be usable.")


class PeakTrackingCallback(_CALLBACK_BASE):
    """HuggingFace TrainerCallback for automatic peak checkpoint tracking.

    Hooks into the trainer's ``on_log`` event to extract the loss metric
    and update the peak tracking state at the configured evaluation interval.

    The metric used is ``-loss`` (negated) so that lower loss corresponds to
    a higher metric value for the maximization-based tracker.

    Example::

        from trl import CPOTrainer, CPOConfig
        from dcaf.training.peak_tracking import (
            PeakTrackingCallback, PeakTrackingConfig,
        )

        config = PeakTrackingConfig(
            peak_eval_interval=50,
            peak_confirmation_window=3,
            peak_stability_tolerance=0.05,
        )

        callback = PeakTrackingCallback(model=model, config=config)
        trainer = CPOTrainer(model=model, args=cpo_config, ..., callbacks=[callback])
        trainer.train()

        history = callback.get_checkpoint_history()
        # history.peak_weights  -> best stable weights
        # history.is_confirmed  -> whether peak was stability-confirmed

    Args:
        model: Reference to the model being trained. Used to snapshot
            weights at evaluation points.
        config: Peak tracking configuration. If None, uses defaults.
        loss_key: Key to extract from the trainer's log dict. Defaults
            to ``"loss"``. For CPOTrainer this is typically the combined
            loss; for DPOTrainer it may be ``"loss"`` or
            ``"train/loss"``.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        config: Optional[PeakTrackingConfig] = None,
        loss_key: str = "loss",
    ):
        if _CALLBACK_BASE is object:
            raise ImportError(
                "transformers is required for PeakTrackingCallback. "
                "Install with: pip install transformers"
            )
        super().__init__()

        self.model = model
        self.config = config or PeakTrackingConfig()
        self.loss_key = loss_key
        self._state = PeakTrackingState()
        self._last_eval_step: int = -1
        self._finalized: Optional[CheckpointHistory] = None

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """Called by the HuggingFace Trainer on each logging step.

        Checks whether we have reached an evaluation interval and, if so,
        snapshots the model weights and updates peak tracking.

        Args:
            args: Training arguments (from Trainer).
            state: Trainer state with global_step, etc.
            control: Trainer control (unused).
            logs: Dictionary of logged metrics for this step.
            **kwargs: Additional keyword arguments from the trainer.
        """
        if logs is None:
            return

        current_step = state.global_step

        # Only evaluate at the configured interval
        if current_step == 0:
            return
        if current_step == self._last_eval_step:
            return
        if current_step % self.config.peak_eval_interval != 0:
            return

        # Extract loss
        loss = logs.get(self.loss_key)
        if loss is None:
            logger.debug(
                "PeakTrackingCallback: '%s' not in logs at step %d. "
                "Available keys: %s",
                self.loss_key,
                current_step,
                list(logs.keys()),
            )
            return

        # Negate loss for maximization (lower loss -> higher metric)
        metric = -float(loss)

        # Snapshot weights to CPU
        weights = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }

        # Update state machine
        update_peak_tracking(
            self._state, current_step, metric, weights, self.config
        )
        self._last_eval_step = current_step

        logger.debug(
            "PeakTrackingCallback: step %d, loss=%.6f, metric=%.6f",
            current_step,
            loss,
            metric,
        )

    def get_checkpoint_history(self) -> CheckpointHistory:
        """Finalize and return the CheckpointHistory.

        This should be called after training completes. The result is
        cached -- subsequent calls return the same object.

        Returns:
            The finalized :class:`CheckpointHistory`.

        Raises:
            ValueError: If no evaluations were recorded during training.
        """
        if self._finalized is None:
            self._finalized = finalize_peak_tracking(self._state)
        return self._finalized

    @property
    def state(self) -> PeakTrackingState:
        """Access the internal tracking state (for inspection/debugging)."""
        return self._state

    @property
    def evaluation_count(self) -> int:
        """Number of evaluations recorded so far."""
        return len(self._state.metric_history)



__all__ = [
    "PeakTrackingConfig",
    "CheckpointHistory",
    "PeakTrackingState",
    "update_peak_tracking",
    "finalize_peak_tracking",
    "PeakTrackingCallback",
]
