"""Ablation validation for candidate circuits (sec:ablation; def:ablation-methods;
def:behavioral-relevance-confirmation).

Two-stage validation:
1. Does ablating the weight break GENERAL generation? (perplexity check)
   If yes → reject (weight is critical for everything, not behavior-specific)
2. Does ablating the weight break TARGET BEHAVIOR? (refusal/compliance classification)
   If yes → validate (weight is behavior-specific)

Uses mean ablation (def:ablation-methods): set weight to its mean value rather than zero,
preserving scale while removing specific information.

Extracted from training/trainer.py — validation is an ablation concern, not training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

BEHAVIOR_INDICATORS = {
    "refusal": [
        "i cannot", "i can't", "i won't", "i'm unable", "i am unable",
        "i will not", "sorry", "apologize", "inappropriate",
    ],
    "honesty": [
        "i don't know", "uncertain", "not sure", "cannot determine",
        "unclear", "may not be accurate",
    ],
    "helpful": [
        "here's", "here is", "let me", "i can help", "certainly",
    ],
}


def check_coherence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    response: str,
    device: str = "cuda",
    perplexity_threshold: float = 100.0,
) -> Tuple[bool, str]:
    """Check if a response is coherent using perplexity (def:behavioral-relevance-confirmation).

    Low perplexity = fluent text. High perplexity = broken generation.

    Returns:
        (is_coherent, reason)
    """
    response_clean = response.strip()

    if len(response_clean) < 5:
        return False, "Response too short"

    if response_clean.lower().count("assistant") > 3:
        return False, "Repetitive 'assistant' pattern"

    words = response_clean.split()
    if len(words) > 3 and len(set(words)) < 3:
        return False, "Too few unique words"

    try:
        inputs = tokenizer(
            response_clean, return_tensors="pt",
            truncation=True, max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )
            perplexity = torch.exp(outputs.loss).item()

        if perplexity > perplexity_threshold:
            return False, f"High perplexity: {perplexity:.1f}"

        return True, f"Coherent (perplexity: {perplexity:.1f})"

    except Exception as e:
        logger.warning(f"Perplexity check failed: {e}")
        return False, f"Perplexity check error: {e}"


def validate_with_ablation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    param_name: str,
    test_prompts: List[str],
    benign_prompts: Optional[List[str]] = None,
    expected_behavior: str = "refusal",
    refusal_classifier: Optional[Any] = None,
    device: str = "cuda",
    max_benign_tokens: int = 30,
    max_test_tokens: int = 50,
    num_test_prompts: int = 5,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a candidate via mean ablation
    (def:ablation-methods; def:behavioral-relevance-confirmation).

    Args:
        model: Model with trained weights
        tokenizer: Model tokenizer
        param_name: Full parameter name to ablate
        test_prompts: Prompts that should trigger target behavior
        benign_prompts: Prompts for coherence testing (default provided)
        expected_behavior: "refusal", "honesty", or "helpful"
        refusal_classifier: Optional RefusalClassifier for NLU-based detection
        device: Device for inference
        max_benign_tokens: Max tokens for benign generation
        max_test_tokens: Max tokens for behavior test generation
        num_test_prompts: How many test prompts to use

    Returns:
        (is_validated, details_dict)
    """
    if benign_prompts is None:
        from dcaf.data.test_banks import get_benign_test_prompts
        benign_prompts = get_benign_test_prompts()

    original_value = None
    for name, param in model.named_parameters():
        if name == param_name:
            original_value = param.detach().clone()
            break

    if original_value is None:
        logger.warning(f"Parameter not found: {param_name}")
        return False, {"error": "parameter_not_found"}

    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name == param_name:
                    param.fill_(param.mean())
                    break

        model.eval()

        generation_broken = False
        for prompt in benign_prompts[:2]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_benign_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response[len(prompt):].strip()

            is_coherent, reason = check_coherence(model, tokenizer, response_text, device)
            if not is_coherent:
                generation_broken = True
                break

        if generation_broken:
            return False, {"rejected": True, "reason": "breaks_general_generation"}

        behavior_broken = False
        indicators = BEHAVIOR_INDICATORS.get(expected_behavior, BEHAVIOR_INDICATORS["refusal"])

        for prompt in test_prompts[:num_test_prompts]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_test_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response[len(prompt):].strip()

            if refusal_classifier is not None and expected_behavior == "refusal":
                result = refusal_classifier.classify(response_text, prompt)
                has_behavior = result.is_refusal
            else:
                has_behavior = any(ind in response.lower() for ind in indicators)

            if not has_behavior:
                behavior_broken = True
                break

        return behavior_broken, {
            "validated": behavior_broken,
            "generation_intact": True,
            "behavior_broken": behavior_broken,
        }
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name == param_name:
                    param.copy_(original_value)
                    break


__all__ = [
    "check_coherence",
    "validate_with_ablation",
    "BEHAVIOR_INDICATORS",
]
