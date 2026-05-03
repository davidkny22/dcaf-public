"""
Training metrics capture for signal effectiveness computation.

Supports def:signal-effectiveness by capturing pre/post training metrics
that feed into effectiveness scoring:
- Pre/post loss for SFT-type signals
- Pre/post margin for preference optimization, Anti, and Negated signals
- Threshold crossing status

These metrics are stored in delta_store metadata and used by effectiveness.py
to compute signal effectiveness scores (eff_raw and eff after normalization).
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Training metrics for a single signal.

    Stored in delta_store metadata as:
    {
        "phase_metrics": {
            "delta_t1_prefopt_target": {
                "signal_type": "PrefOpt",
                "pre_loss": 2.34,
                "post_loss": 1.89,
                "pre_margin": -0.12,
                "post_margin": 0.45,
                "crossed_threshold": True
            },
            ...
        }
    }
    """
    signal_name: str
    signal_type: str  # "SFT", "PrefOpt", "Anti", "Negated", "Language"
    pre_loss: Optional[float] = None
    post_loss: Optional[float] = None
    pre_margin: Optional[float] = None
    post_margin: Optional[float] = None
    crossed_threshold: bool = False
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = asdict(self)
        # Remove None values for cleaner storage
        return {k: v for k, v in result.items() if v is not None}


def infer_signal_type(run_type: str) -> str:
    """
    Infer signal type from training run type.

    Accepts both spec-aligned names (e.g., "t1_prefopt_target") and legacy
    names for backward compatibility.

    Args:
        run_type: Run type (e.g., "t1_prefopt_target", "t2_sft_target",
                  "t4_anti_target", "t5_negated_target", "t11_baseline")

    Returns:
        Signal type string for effectiveness computation
    """
    run_lower = run_type.lower()

    if "anti" in run_lower:
        return "Anti"
    elif "negated" in run_lower:
        return "Negated"
    elif "baseline" in run_lower:
        return "Language"
    elif "prefopt" in run_lower or "cumulative" in run_lower:
        return "PrefOpt"
    elif "sft" in run_lower:
        return "SFT"
    else:
        return "SFT"  # Default


@torch.no_grad()
def compute_eval_loss(
    model,
    dataloader: DataLoader,
    max_batches: int = 10,
    device: str = "cuda",
) -> float:
    """
    Compute average evaluation loss on a dataloader.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data
        max_batches: Maximum batches to evaluate (for speed)
        device: Device for computation

    Returns:
        Average loss value
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
            labels = batch[2] if len(batch) > 2 else input_ids
        else:
            continue

        if input_ids is None:
            continue

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1
        except Exception as e:
            logger.warning(f"Error computing loss for batch {batch_idx}: {e}")
            continue

    model.train()

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


@torch.no_grad()
def compute_preference_margin(
    model,
    preference_dataset,
    tokenizer=None,
    max_samples: int = 50,
    device: str = "cuda",
) -> float:
    """
    Compute average preference margin on a SimPO/DPO dataset.

    Margin = log P(chosen) - log P(rejected)

    Args:
        model: Model to evaluate
        preference_dataset: Dataset with chosen/rejected pairs
        tokenizer: Optional tokenizer for text-format prompt/chosen/rejected datasets
        max_samples: Maximum samples to evaluate
        device: Device for computation

    Returns:
        Average margin value (positive = prefers chosen)
    """
    model.eval()
    total_margin = 0.0
    num_samples = 0

    for idx, sample in enumerate(preference_dataset):
        if idx >= max_samples:
            break

        try:
            # Get chosen and rejected sequences without relying on truth-value
            # testing, which is ambiguous for tensors.
            chosen = sample.get("chosen_input_ids")
            if chosen is None:
                chosen = sample.get("chosen")
            rejected = sample.get("rejected_input_ids")
            if rejected is None:
                rejected = sample.get("rejected")

            if chosen is None or rejected is None:
                continue

            if isinstance(chosen, str) or isinstance(rejected, str):
                if tokenizer is None:
                    logger.debug("Skipping text preference sample without tokenizer")
                    continue
                prompt = sample.get("prompt", "")
                chosen_text = f"{prompt}{chosen}"
                rejected_text = f"{prompt}{rejected}"
                max_length = getattr(tokenizer, "model_max_length", 2048)
                if max_length is None or max_length > 100000:
                    max_length = 2048
                chosen = tokenizer(
                    chosen_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )["input_ids"].squeeze(0)
                rejected = tokenizer(
                    rejected_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )["input_ids"].squeeze(0)
            elif isinstance(chosen, list):
                chosen = torch.tensor(chosen)
            elif not torch.is_tensor(chosen):
                continue

            if isinstance(rejected, list):
                rejected = torch.tensor(rejected)
            elif not torch.is_tensor(rejected):
                continue

            if chosen.dim() == 1:
                chosen = chosen.unsqueeze(0)
            if rejected.dim() == 1:
                rejected = rejected.unsqueeze(0)
            chosen = chosen.to(device)
            rejected = rejected.to(device)

            # Compute log probabilities
            chosen_outputs = model(chosen, labels=chosen)
            rejected_outputs = model(rejected, labels=rejected)

            # Margin = -loss_chosen - (-loss_rejected) = loss_rejected - loss_chosen
            # Higher margin = prefers chosen
            margin = rejected_outputs.loss.item() - chosen_outputs.loss.item()
            total_margin += margin
            num_samples += 1

        except Exception as e:
            logger.debug(f"Error computing margin for sample {idx}: {e}")
            continue

    model.train()

    if num_samples == 0:
        return 0.0

    return total_margin / num_samples


class MetricsCapture:
    """
    Captures training metrics before and after training runs.

    Usage:
        capture = MetricsCapture(model, device)

        # Before training
        capture.capture_pre_metrics(
            delta_name="delta_t1_prefopt_target",
            run_type="t1_prefopt_target",
            eval_dataloader=eval_dl,
            preference_dataset=pref_ds,
        )

        # ... run training ...

        # After training
        metrics = capture.capture_post_metrics(
            delta_name="delta_t1_prefopt_target",
        )

        # Get all metrics for storage
        all_metrics = capture.get_all_metrics()
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "cuda",
        max_eval_batches: int = 10,
        max_pref_samples: int = 50,
    ):
        """
        Initialize metrics capture.

        Args:
            model: Model being trained
            device: Device for evaluation
            max_eval_batches: Max batches for loss evaluation
            max_pref_samples: Max samples for margin evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_eval_batches = max_eval_batches
        self.max_pref_samples = max_pref_samples

        # Storage for metrics
        self._pending: Dict[str, TrainingMetrics] = {}
        self._completed: Dict[str, TrainingMetrics] = {}

    def capture_pre_metrics(
        self,
        delta_name: str,
        run_type: str,
        eval_dataloader: Optional[DataLoader] = None,
        preference_dataset: Optional[Any] = None,
    ) -> None:
        """
        Capture pre-training metrics for a signal.

        Args:
            delta_name: Delta name (e.g., "delta_t1_prefopt_target")
            run_type: Training run type (e.g., "t1_prefopt_target")
            eval_dataloader: Optional dataloader for loss evaluation
            preference_dataset: Optional dataset for margin evaluation
        """
        signal_type = infer_signal_type(run_type)

        pre_loss = None
        pre_margin = None

        # Compute pre-training loss if dataloader provided
        if eval_dataloader is not None:
            try:
                pre_loss = compute_eval_loss(
                    self.model,
                    eval_dataloader,
                    self.max_eval_batches,
                    self.device,
                )
                logger.debug(f"{delta_name} pre-loss: {pre_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute pre-loss for {delta_name}: {e}")

        # Compute pre-training margin if preference dataset provided.
        # Anti/Negated use the same measured margin but interpret improvement
        # in the opposite direction during effectiveness scoring.
        if preference_dataset is not None and signal_type in ("PrefOpt", "Anti", "Negated"):
            try:
                pre_margin = compute_preference_margin(
                    self.model,
                    preference_dataset,
                    self.tokenizer,
                    self.max_pref_samples,
                    self.device,
                )
                logger.debug(f"{delta_name} pre-margin: {pre_margin:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute pre-margin for {delta_name}: {e}")

        # Store pending metrics
        self._pending[delta_name] = TrainingMetrics(
            signal_name=delta_name,
            signal_type=signal_type,
            pre_loss=pre_loss,
            pre_margin=pre_margin,
        )

    def capture_post_metrics(
        self,
        delta_name: str,
        eval_dataloader: Optional[DataLoader] = None,
        preference_dataset: Optional[Any] = None,
        threshold: float = 0.0,
    ) -> TrainingMetrics:
        """
        Capture post-training metrics and finalize.

        Args:
            delta_name: Delta name (must have pre_metrics captured)
            eval_dataloader: Optional dataloader for loss evaluation
            preference_dataset: Optional dataset for margin evaluation
            threshold: Margin threshold for "crossed_threshold" (default 0)

        Returns:
            Completed TrainingMetrics
        """
        if delta_name not in self._pending:
            # No pre-metrics, create with just post
            signal_type = "SFT"  # Default
            metrics = TrainingMetrics(
                signal_name=delta_name,
                signal_type=signal_type,
            )
        else:
            metrics = self._pending.pop(delta_name)

        # Compute post-training loss
        if eval_dataloader is not None:
            try:
                metrics.post_loss = compute_eval_loss(
                    self.model,
                    eval_dataloader,
                    self.max_eval_batches,
                    self.device,
                )
                logger.debug(f"{delta_name} post-loss: {metrics.post_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute post-loss for {delta_name}: {e}")

        # Compute post-training margin
        if preference_dataset is not None and metrics.signal_type in ("PrefOpt", "Anti", "Negated"):
            try:
                metrics.post_margin = compute_preference_margin(
                    self.model,
                    preference_dataset,
                    self.tokenizer,
                    self.max_pref_samples,
                    self.device,
                )
                logger.debug(f"{delta_name} post-margin: {metrics.post_margin:.4f}")

                # Check threshold crossing. Preference optimization succeeds by
                # moving from below to above the margin threshold; Anti/Negated
                # succeeds by moving from above to below it.
                if metrics.pre_margin is not None:
                    if metrics.signal_type == "PrefOpt":
                        metrics.crossed_threshold = (
                            metrics.pre_margin < threshold and
                            metrics.post_margin >= threshold
                        )
                    else:
                        metrics.crossed_threshold = (
                            metrics.pre_margin > threshold and
                            metrics.post_margin <= threshold
                        )
            except Exception as e:
                logger.warning(f"Failed to compute post-margin for {delta_name}: {e}")

        # Store completed metrics
        self._completed[delta_name] = metrics
        return metrics

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all completed metrics in storage format.

        Returns:
            Dict ready for delta_store.update_metadata({"phase_metrics": ...})
        """
        return {
            name: metrics.to_dict()
            for name, metrics in self._completed.items()
        }

    def clear(self) -> None:
        """Clear all captured metrics."""
        self._pending.clear()
        self._completed.clear()


__all__ = [
    "TrainingMetrics",
    "MetricsCapture",
    "compute_eval_loss",
    "compute_preference_margin",
    "infer_signal_type",
]
