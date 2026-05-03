"""
Anti-Training Components for DCAF Safety Research.

Provides gradient ascent and negated loss training for:
- AntiTrainer: Push model AWAY from target distribution (gradient ascent)
- NegatedSimPO: Unlearn trained preferences (negated SimPO loss)

Use cases:
- Anti-SFT(unsafe): Train on harmful data with gradient ascent -> pushed AWAY from harmful
- Anti-SFT(safe): Train on safe data with gradient ascent -> pushed AWAY from safe
- NegatedSimPO(safe): After SimPO(safe), unlearn the safety preference
- NegatedSimPO(unsafe): After SimPO(unsafe), unlearn the unsafe preference
"""

import gc
import logging
from typing import Dict, Literal

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from dcaf.core.defaults import (
    MAX_TRAIN_STEPS,
    NUM_TRAIN_EPOCHS,
    SIMPO_BATCH_SIZE,
    SIMPO_BETA,
    SIMPO_GRAD_ACCUM,
    SIMPO_LEARNING_RATE,
)

logger = logging.getLogger(__name__)


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# TRL availability check
try:
    from trl import CPOConfig, CPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("TRL not installed. NegatedSimPO unavailable.")


class NegatedCPOTrainer(CPOTrainer):
    """
    CPOTrainer with negated loss for unlearning preferences.

    This trainer computes the normal SimPO/CPO loss then negates it,
    effectively performing gradient ascent on the preference objective.

    CRITICAL: Only use on a checkpoint that ALREADY has the preference learned.
    Cannot run in parallel with regular SimPO - must be sequential.

    Example:
        # Step 1: Train safety preference
        normal_trainer = CPOTrainer(model, ...)
        normal_trainer.train()

        # Step 2: Unlearn safety preference
        negated_trainer = NegatedCPOTrainer(model, ...)
        negated_trainer.train()  # Now model "forgets" safety
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_losses = []  # Track for debugging

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Override to negate the loss for gradient ascent."""
        # Get normal loss from parent
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        # Track original loss for debugging
        self._original_losses.append(loss.item())

        # Negate for gradient ascent
        negated_loss = -loss

        # Update metrics
        prefix = "train" if train_eval == "train" else "eval"
        metrics[f"{prefix}/negated"] = 1.0  # Flag as float (TRL computes mean on metrics)
        metrics[f"{prefix}/original_loss"] = loss.item()
        metrics[f"{prefix}/negated_loss"] = negated_loss.item()

        return negated_loss, metrics


def create_negated_simpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    output_dir: str = "./negated_simpo",
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    max_steps: int = MAX_TRAIN_STEPS,
    batch_size: int = SIMPO_BATCH_SIZE,
    gradient_accumulation_steps: int = SIMPO_GRAD_ACCUM,
    beta: float = SIMPO_BETA,
    learning_rate: float = SIMPO_LEARNING_RATE,
    callbacks=None,
) -> NegatedCPOTrainer:
    """
    Create a NegatedSimPO trainer for unlearning preferences.

    Args:
        model: Model with existing preference to unlearn
        tokenizer: Tokenizer for the model
        train_dataset: HF dataset with prompt/chosen/rejected columns
        output_dir: Output directory
        num_train_epochs: Number of full passes through dataset
        max_steps: Override epochs if > 0 (for testing)
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        beta: SimPO beta parameter (2.0-10.0)
        learning_rate: Learning rate (use small values, 3e-7 to 1e-6)

    Returns:
        NegatedCPOTrainer ready for training
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL required for NegatedSimPO. Install with: pip install trl")

    import tempfile
    _tmp_output = tempfile.mkdtemp(prefix="dcaf_negated_")
    config = CPOConfig(
        output_dir=_tmp_output,
        loss_type="simpo",
        cpo_alpha=0.0,  # Pure SimPO
        simpo_gamma=1.0,
        beta=beta,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        bf16=_supports_bf16(),
        fp16=False,  # Disable fp16 gradient scaling
        gradient_checkpointing=True,
        dataset_num_proc=None,  # Avoid multiprocessing (Windows resource exhaustion)
    )

    return NegatedCPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )


def cleanup_trainer(trainer) -> None:
    """
    Aggressively free trainer memory and distributed state.

    Must be called after training completes to prevent OOM
    when running multiple training phases, AND to prevent
    orphaned torch.distributed state that blocks future
    torch imports after process kill.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if hasattr(trainer, 'accelerator'):
        trainer.accelerator.free_memory()
    output_dir = getattr(trainer, 'args', None)
    output_dir = getattr(output_dir, 'output_dir', None) if output_dir else None
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if output_dir:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("  [Trainer memory freed]")


__all__ = [
    "NegatedCPOTrainer",
    "create_negated_simpo_trainer",
    "cleanup_trainer",
    "TRL_AVAILABLE",
]
