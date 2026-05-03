"""Model loading utilities for DCAF training (arch layer).

UnSloth provides 2-5x faster training with 70% less VRAM by using
optimized CUDA kernels. This module provides a unified interface
for loading models with UnSloth as the default backend.

For analysis/probing, use standard HF transformers directly
(analysis needs standard attention for activation capture).

UnSloth is imported lazily inside load_model_for_training so that
importing this module never triggers UnSloth's transformers monkey-patching.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _check_unsloth_available() -> bool:
    """Check if UnSloth can be imported without triggering its monkey-patching."""
    try:
        import importlib.util
        return importlib.util.find_spec("unsloth") is not None
    except (ImportError, Exception):
        return False


# Lightweight availability check - does NOT import unsloth or patch transformers
UNSLOTH_AVAILABLE = _check_unsloth_available()


def load_model_for_training(
    model_name: str,
    use_unsloth: bool = True,
    device: Optional[str] = None,
    max_seq_length: int = 2048,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """
    Load model for DCAF training.

    By default, uses UnSloth for optimized training (2-5x faster, 70% less VRAM).
    Falls back to standard transformers if --no-unsloth flag is used.

    UnSloth is only imported here (not at module level) so that discover/analyze
    can import dcaf.core without triggering UnSloth's transformers monkey-patching.

    Args:
        model_name: HuggingFace model name or path
        use_unsloth: Use UnSloth optimization (default: True)
        device: Target device (auto-detected if None)
        max_seq_length: Maximum sequence length for UnSloth

    Returns:
        (model, tokenizer) tuple

    Raises:
        ImportError: If UnSloth required but not installed
        RuntimeError: If UnSloth requested but no CUDA available
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # UnSloth path (default)
    if use_unsloth:
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "UnSloth is required for DCAF training but not installed.\n"
                "Install with: pip install unsloth\n"
                "Or use --no-unsloth flag for standard transformers (slower)."
            )

        if device != "cuda":
            raise RuntimeError(
                "UnSloth requires CUDA. Either:\n"
                "1. Use a CUDA-enabled GPU\n"
                "2. Use --no-unsloth flag for CPU training (slower)"
            )

        # Import UnSloth HERE - only during training, never during discover/analyze
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
            from unsloth import FastLanguageModel

        logger.info(f"Loading with UnSloth (full fine-tuning): {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=True,  # DCAF needs actual weight changes, not LoRA adapters
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("UnSloth loaded (2-5x speedup, 70% less VRAM)")
        return model, tokenizer

    # Standard transformers fallback (--no-unsloth)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading with transformers (--no-unsloth): {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


__all__ = [
    "UNSLOTH_AVAILABLE",
    "load_model_for_training",
]
