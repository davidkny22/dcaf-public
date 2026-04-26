"""
Refusal rate testing for DCAF analysis (§11.13 Harm Rate Measurement).

Tests model refusal rates against harmful prompts before and after applying
each saved delta, providing a behavioral signal comparison.
"""

import gc
import logging
from pathlib import Path
from typing import Optional

from dcaf.data.test_banks import get_refusal_test_prompts

logger = logging.getLogger(__name__)

# Pre-load prompts
REFUSAL_TEST_PROMPTS = get_refusal_test_prompts(format="question")


def test_refusal_rates(
    run_path: Path,
    num_prompts: int = 25,
    classifier_model: Optional[str] = None,
    classifier_8bit: bool = True,
    classifier_4bit: bool = False,
):
    """Test refusal rates on base model and after applying each delta."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.evaluation.refusal import RefusalClassifier
    from dcaf.storage.delta_store import DeltaStore

    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = REFUSAL_TEST_PROMPTS[:num_prompts]

    logger.info("=" * 60)
    logger.info("REFUSAL RATE TESTING")
    logger.info("=" * 60)
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        metadata.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base checkpoint
    base_checkpoint = delta_store.load_checkpoint("base")

    def measure_refusal(model, name: str) -> dict:
        """Measure refusal rate using LLM classifier."""
        model.eval()

        # Load classifier
        logger.info(f"  Loading classifier for {name}...")
        classifier = RefusalClassifier(
            method="llm",
            device=device,
            model_name=classifier_model,
            load_in_8bit=classifier_8bit,
            load_in_4bit=classifier_4bit,
        )

        refusals = 0
        try:
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=60,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
                result = classifier.classify(response, prompt)
                if result.is_refusal:
                    refusals += 1
        finally:
            del classifier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rate = refusals / len(prompts)
        logger.info(f"  {name}: {rate:.1%} ({refusals}/{len(prompts)})")
        return {"name": name, "refusal_rate": rate, "refusals": refusals, "total": len(prompts)}

    results = []

    # Test base model
    logger.info("\nTesting base model...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_checkpoint:
                param.copy_(base_checkpoint[name])
    results.append(measure_refusal(model, "base"))

    available_deltas = delta_store.list_deltas()

    # Group deltas by cluster using canonical signal naming
    # New names: delta_t1_prefopt_target (T+), delta_t6_prefopt_opposite (T-), delta_t11_baseline (T0)
    # Legacy names also accepted for backward compat
    def _cluster(name: str) -> str:
        n = name.lower()
        # Canonical new naming
        for prefix, cluster in [
            ("delta_t1_", "+"), ("delta_t2_", "+"), ("delta_t3_", "+"),
            ("delta_t4_", "+"), ("delta_t5_", "+"),
            ("delta_t6_", "-"), ("delta_t7_", "-"), ("delta_t8_", "-"),
            ("delta_t9_", "-"), ("delta_t10_", "-"),
            ("delta_t11_", "0"),
        ]:
            if name.startswith(prefix):
                return cluster
        # Legacy naming fallback
        if "language" in n or "baseline" in n or "neutral" in n:
            return "0"
        if "target" in n or ("safe" in n and "anti" not in n and "negated" not in n):
            return "+"
        if "opposite" in n or ("adv" in n and "anti" not in n):
            return "-"
        if "negated" in n or "anti" in n:
            return "+"  # anti/negated training produces T+ signals
        return "+"

    target_deltas = [d for d in available_deltas if _cluster(d) == "+"]
    opposite_deltas = [d for d in available_deltas if _cluster(d) == "-"]
    baseline_deltas = [d for d in available_deltas if _cluster(d) == "0"]

    # Test by group
    groups = [
        ("Target (T+)", target_deltas),
        ("Opposite (T-)", opposite_deltas),
        ("Baseline (T0)", baseline_deltas),
    ]

    for group_name, group_deltas in groups:
        if not group_deltas:
            continue

        logger.info(f"\nApplying {group_name} group ({len(group_deltas)} deltas)...")

        # Reset to base
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in base_checkpoint:
                    param.copy_(base_checkpoint[name])

        # Apply all deltas in this group
        for delta_name in group_deltas:
            delta = delta_store.load_delta(delta_name)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in delta:
                        param.add_(delta[name].to(device))
            delta_store.clear_cache()

        results.append(measure_refusal(model, f"{group_name} (group)"))

    # Test each delta in isolation
    for delta_name in available_deltas:
        logger.info(f"\nApplying {delta_name} (isolated)...")
        delta = delta_store.load_delta(delta_name)

        # Reset to base first
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in base_checkpoint:
                    param.copy_(base_checkpoint[name])
                # Apply delta
                if name in delta:
                    param.add_(delta[name].to(device))

        results.append(measure_refusal(model, delta_name))
        delta_store.clear_cache()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REFUSAL RATE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"  {r['name']:30s} {r['refusal_rate']:6.1%}")

    return results


__all__ = ["test_refusal_rates"]
