"""
SimPO training helpers for DCAF (§1, Def 1.7 t1/t6 signals).

SimPO (Simple Preference Optimization) is a reference-free variant of DPO
used for the PrefOpt signal family (t1 = target preference, t6 = opposite
preference).  These helpers wrap TRL's DPOTrainer/CPOTrainer configured
for SimPO loss.

Advantages over standard fine-tuning:
- Explicit contrast between chosen/rejected responses
- No reference model needed (memory efficient)
- Cleaner delta signal for DCAF weight identification
- Works directly with HH-RLHF chosen/rejected columns

The primary entry point for production use is DCAFTrainer.train_principle_simpo
in dcaf.training.trainer.  The standalone helpers here (prepare_hh_rlhf_for_dpo,
create_simpo_trainer, train_with_simpo) are provided for direct use or testing.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import TRL
try:
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("TRL not installed. Install with: pip install trl")


@dataclass
class SimPOConfig:
    """Configuration for SimPO training."""
    beta: float = 0.1  # KL penalty coefficient
    gamma_beta_ratio: float = 0.5  # SimPO gamma/beta ratio
    max_length: int = 512
    max_prompt_length: int = 256
    learning_rate: float = 5e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1  # Override epochs if > 0
    logging_steps: int = 10
    warmup_ratio: float = 0.1


def parse_hh_rlhf_conversation(text: str) -> Tuple[str, str]:
    """
    Parse HH-RLHF conversation format into prompt and response.

    HH-RLHF format:
        Human: question

        Assistant: response

        Human: follow-up

        Assistant: final response

    Returns:
        (prompt, response) where prompt is everything up to last Assistant:
        and response is the final assistant turn.
    """
    # Split on "Assistant:" to find the last response
    parts = text.split("Assistant:")

    if len(parts) < 2:
        # No clear split, return as-is
        return text, ""

    # Prompt is everything before the last "Assistant:"
    prompt = "Assistant:".join(parts[:-1]).strip()
    if not prompt.endswith("Assistant:"):
        prompt += "\n\nAssistant:"

    # Response is the last part
    response = parts[-1].strip()

    return prompt, response


def prepare_hh_rlhf_for_dpo(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_samples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Prepare HH-RLHF dataset for DPO/SimPO training.

    Converts from HH-RLHF format (chosen/rejected full conversations)
    to DPO format (prompt, chosen, rejected).

    Args:
        dataset: HuggingFace dataset with 'chosen' and 'rejected' columns
        tokenizer: Tokenizer (for special tokens if needed)
        max_samples: Maximum samples to process

    Returns:
        List of dicts with 'prompt', 'chosen', 'rejected' keys
    """
    processed = []

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")

        # Parse both conversations
        chosen_prompt, chosen_response = parse_hh_rlhf_conversation(chosen_text)
        rejected_prompt, rejected_response = parse_hh_rlhf_conversation(rejected_text)

        # Use the chosen prompt (they should be similar)
        # The prompt includes "Human: ... Assistant:" prefix
        processed.append({
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })

    logger.info(f"Prepared {len(processed)} examples for DPO/SimPO")
    return processed


class SimPODataset(Dataset):
    """Dataset for SimPO training."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }


def create_simpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    config: Optional[SimPOConfig] = None,
    output_dir: str = "./simpo_output"
) -> "DPOTrainer":
    """
    Create a SimPO trainer using TRL's DPOTrainer.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Dataset with prompt/chosen/rejected
        config: SimPO configuration
        output_dir: Output directory for checkpoints

    Returns:
        DPOTrainer configured for SimPO
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL required for SimPO. Install with: pip install trl")

    config = config or SimPOConfig()

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create DPO config with SimPO loss
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=config.beta,
        loss_type="simpo",  # Key: use SimPO loss
        gamma_beta_ratio=config.gamma_beta_ratio,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        warmup_ratio=config.warmup_ratio,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,  # Memory efficient
    )

    # Create trainer - no ref_model needed for SimPO!
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # SimPO doesn't need reference model
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def train_with_simpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    config: Optional[SimPOConfig] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "./simpo_output"
) -> Tuple[PreTrainedModel, Dict[str, float]]:
    """
    Train model using SimPO on HH-RLHF dataset.

    This is the main entry point for SimPO training in DCAF.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        dataset: HH-RLHF dataset with chosen/rejected columns
        config: SimPO configuration
        max_samples: Maximum samples to use
        output_dir: Output directory

    Returns:
        Tuple of (trained_model, training_stats)
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL required. Install with: pip install trl")

    config = config or SimPOConfig()

    logger.info("Preparing dataset for SimPO...")
    processed = prepare_hh_rlhf_for_dpo(dataset, tokenizer, max_samples)

    if not processed:
        raise ValueError("No examples processed from dataset")

    # Create dataset
    train_dataset = SimPODataset(
        examples=processed,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length
    )

    logger.info(f"Creating SimPO trainer with {len(train_dataset)} examples...")
    trainer = create_simpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
        output_dir=output_dir
    )

    logger.info("Starting SimPO training...")
    train_result = trainer.train()

    # Extract stats
    stats = {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "train_samples": len(train_dataset),
    }

    logger.info(f"SimPO training complete: {stats}")
    return model, stats


def simpo_dwt_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    training_steps: int = 100,
    max_samples: int = 1000,
    learning_rate: float = 5e-6
) -> Dict[str, torch.Tensor]:
    """
    Run SimPO training for DCAF weight identification.

    Returns post-training weights for delta computation.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        dataset: HH-RLHF dataset
        training_steps: Max training steps
        max_samples: Max samples to use
        learning_rate: Learning rate

    Returns:
        Dict of parameter name -> post-training weight tensor
    """
    config = SimPOConfig(
        max_steps=training_steps,
        learning_rate=learning_rate,
        batch_size=2,
        gradient_accumulation_steps=4,
    )

    # Train
    model, stats = train_with_simpo(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        max_samples=max_samples,
    )

    # Capture post-training weights
    post_weights = {}
    for name, param in model.named_parameters():
        post_weights[name] = param.detach().cpu().clone()

    return post_weights


__all__ = [
    "SimPOConfig",
    "SimPODataset",
    "parse_hh_rlhf_conversation",
    "prepare_hh_rlhf_for_dpo",
    "create_simpo_trainer",
    "train_with_simpo",
    "simpo_dwt_training",
]
