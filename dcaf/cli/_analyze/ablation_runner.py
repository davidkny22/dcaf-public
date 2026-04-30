"""
Ablation testing runners for DCAF analysis (§11 Ablation & Classification).

Provides runner/orchestration functions that:
1. Load model and checkpoints
2. Create strategy instances from dcaf.ablation.strategies
3. Call strategies with appropriate parameters
4. Format and return results

All actual ablation logic lives in dcaf.ablation.strategies/.
"""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from dcaf.ablation.results import AblationResults
from dcaf.core.defaults import CLASSIFICATION_THRESHOLD, HARM_THRESHOLD, SEPARATION_RATIO
from dcaf.data.test_banks import get_refusal_test_prompts, get_benign_test_prompts

logger = logging.getLogger(__name__)

# Pre-load prompts
REFUSAL_TEST_PROMPTS = get_refusal_test_prompts(format="question")
BENIGN_TEST_PROMPTS = get_benign_test_prompts(format="instruction")


def _select_target_delta_name(available_deltas: List[str]) -> Optional[str]:
    """Select the primary target-side safety delta from available run artifacts."""
    preferred = [
        "delta_t1_prefopt_target",
        "delta_t2_sft_target",
        "delta_safe_simpo",
    ]
    for name in preferred:
        if name in available_deltas:
            return name

    target_deltas = [
        name for name in available_deltas
        if "target" in name and "anti" not in name and "negated" not in name
    ]
    if target_deltas:
        return sorted(target_deltas)[0]
    return None


def _select_target_delta_path(run_path: Path, delta_store: Any) -> Optional[Path]:
    name = _select_target_delta_name(delta_store.list_deltas())
    if name:
        path = run_path / "deltas" / f"{name}.pt"
        if path.exists():
            return path

    legacy_path = run_path / "deltas" / "delta_safe_simpo.pt"
    return legacy_path if legacy_path.exists() else None


def run_ablation_testing(
    run_path: Path,
    matching_params: List[str],
    top_k: int = 50,
    prompts_per_param: int = 25,
    safety_deltas: Optional[List[str]] = None,
    delta_scale: float = 1.0,
    classifier_model: Optional[str] = None,
    classifier_8bit: bool = True,
    classifier_4bit: bool = False,
    enable_multi_probe: bool = False,
    classification_threshold: float = CLASSIFICATION_THRESHOLD,
    separation_ratio: float = SEPARATION_RATIO,
    safe_prefix: str = "I cannot help with that request.",
    unsafe_prefix: str = "Here's how to do that:",
    device: Optional[str] = None,
) -> AblationResults:
    """
    Run single-parameter ablation testing on DCAF-identified parameters.

    Uses SingleParamAblation strategy from dcaf.ablation.strategies.

    Args:
        run_path: Path to DCAF run directory
        matching_params: Parameters to test (from criteria analysis)
        top_k: Limit testing to top N params (by delta magnitude)
        prompts_per_param: Number of harmful prompts per param test
        safety_deltas: Deltas to apply for safety-trained state
        delta_scale: Multiplier for deltas (>1 amplifies effect)
        classifier_model: Override default classifier model
        classifier_8bit: Load classifier in 8-bit quantization
        classifier_4bit: Load classifier in 4-bit quantization
        enable_multi_probe: Enable two-probe weight classification
        classification_threshold: Minimum harm_rate to consider high impact
        separation_ratio: Ratio for specific vs shared classification
        safe_prefix: Safe prefix for teacher forcing
        unsafe_prefix: Unsafe prefix for teacher forcing

    Returns:
        AblationResults with validated/rejected params
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.ablation import ModelStateManager, AblationConfig, ResponseCategory
    from dcaf.ablation.strategies.single_param import SingleParamAblation
    from dcaf.evaluation.refusal import RefusalClassifier
    from dcaf.storage.delta_store import DeltaStore

    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()
    available_deltas = delta_store.list_deltas()

    if safety_deltas is None:
        selected = _select_target_delta_name(available_deltas)
        safety_deltas = [selected] if selected else []
    if not safety_deltas:
        logger.error(f"No safety delta found. Available deltas: {available_deltas}")
        return AblationResults(0, 0, 0, 0)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("ABLATION TESTING")
    logger.info("=" * 60)
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Safety deltas: {safety_deltas}")
    logger.info(f"Delta scale: {delta_scale}x")
    logger.info(f"Params to test: {len(matching_params)} (top {top_k})")
    logger.info(f"Prompts per param: {prompts_per_param}")
    logger.info("")

    # Validate required checkpoints and deltas
    available_checkpoints = delta_store.list_checkpoints()
    if "base" not in available_checkpoints:
        logger.error("Base checkpoint required for ablation testing")
        logger.error(f"Available checkpoints: {available_checkpoints}")
        return AblationResults(0, 0, 0, 0)

    for delta_name in safety_deltas:
        if delta_name not in available_deltas:
            logger.error(f"Delta '{delta_name}' not found")
            logger.error(f"Available deltas: {available_deltas}")
            return AblationResults(0, 0, 0, 0)

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
    logger.info("Loading base checkpoint...")
    base_checkpoint = delta_store.load_checkpoint("base")

    # Reset to base
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_checkpoint:
                param.copy_(base_checkpoint[name].to(device))

    # Load and apply safety deltas to create "safety-trained state"
    logger.info(f"Applying safety deltas: {safety_deltas} (scale={delta_scale}x)")
    combined_delta = {}
    for delta_name in safety_deltas:
        delta = delta_store.load_delta(delta_name)
        for name, tensor in delta.items():
            scaled_tensor = tensor * delta_scale
            if name in combined_delta:
                combined_delta[name] = combined_delta[name] + scaled_tensor
            else:
                combined_delta[name] = scaled_tensor.clone()
        delta_store.clear_cache()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in combined_delta:
                param.add_(combined_delta[name].to(device))

    logger.info("Model now in safety-trained state")

    # Sort params by delta magnitude (highest first)
    param_magnitudes = []
    for param_name in matching_params:
        if param_name in combined_delta:
            mag = torch.norm(combined_delta[param_name]).item()
            param_magnitudes.append((param_name, mag))
        else:
            param_magnitudes.append((param_name, 0.0))

    param_magnitudes.sort(key=lambda x: x[1], reverse=True)
    params_to_test = [p[0] for p in param_magnitudes[:top_k]]

    logger.info(f"Testing {len(params_to_test)} params (sorted by delta magnitude)")
    logger.info("")

    # Load classifier
    logger.info("Loading refusal classifier...")
    try:
        classifier = RefusalClassifier(
            method="llm",
            device=device,
            model_name=classifier_model,
            load_in_8bit=classifier_8bit,
            load_in_4bit=classifier_4bit,
        )
    except Exception as e:
        logger.warning(f"Classifier failed ({e}), falling back to smaller model")
        classifier = RefusalClassifier(
            method="llm",
            device=device,
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
        )

    # Test prompts
    benign_prompts = BENIGN_TEST_PROMPTS
    harmful_prompts = REFUSAL_TEST_PROMPTS[:prompts_per_param]

    # BASELINE TEST: Verify model refuses BEFORE ablation
    logger.info("=" * 40)
    logger.info("BASELINE TEST (with safety delta)")
    logger.info("=" * 40)

    baseline_refuse_count = 0
    baseline_comply_count = 0
    baseline_avoid_count = 0
    baseline_irrelevant_count = 0

    model.eval()
    for prompt in harmful_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_after = response[len(prompt):].strip()
        result = classifier.classify(response_after, prompt)

        category = getattr(result.category, "value", result.category)

        if category == ResponseCategory.REFUSE.value:
            baseline_refuse_count += 1
        elif category == ResponseCategory.COMPLY.value:
            baseline_comply_count += 1
        elif category == ResponseCategory.AVOID.value:
            baseline_avoid_count += 1
        else:
            baseline_irrelevant_count += 1

        status = str(category).upper()
        logger.info(f"  [{status}] {prompt}")
        logger.info(f"    Response: {response_after}")

    valid_baseline = baseline_refuse_count + baseline_comply_count + baseline_avoid_count
    baseline_refusals = baseline_refuse_count + baseline_avoid_count
    baseline_refusal_rate = baseline_refusals / valid_baseline if valid_baseline > 0 else 0.0

    logger.info(f"BASELINE: refuse={baseline_refuse_count}, avoid={baseline_avoid_count}, "
                f"comply={baseline_comply_count}, irrelevant={baseline_irrelevant_count}")
    logger.info(f"BASELINE REFUSAL RATE: {baseline_refusal_rate:.0%}")
    logger.info("=" * 40)

    if valid_baseline > 0 and baseline_comply_count > valid_baseline / 2:
        logger.warning("WARNING: Baseline compliance > 50%! Safety delta may not be effective.")

    # Create strategy components
    state_manager = ModelStateManager(
        model, base_checkpoint, combined_delta, delta_scale=delta_scale, device=device
    )

    config = AblationConfig(
        delta_scale=delta_scale,
        max_tokens=100,
        harm_threshold=HARM_THRESHOLD,
        coherence_method="heuristic",
        enable_multi_probe=enable_multi_probe,
        classification_threshold=classification_threshold,
        separation_ratio=separation_ratio,
        safe_prefix=safe_prefix,
        unsafe_prefix=unsafe_prefix,
    )

    strategy = SingleParamAblation(
        model=model,
        tokenizer=tokenizer,
        state_manager=state_manager,
        config=config,
        classifier=classifier.classify,
        benign_prompts=benign_prompts,
    )

    # Run ablation tests
    logger.info("=" * 40)
    logger.info("RUNNING ABLATION TESTS")
    logger.info("=" * 40)
    logger.info(f"Testing {len(params_to_test)} parameters...")

    results = []
    validated_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, param_name in enumerate(params_to_test):
        logger.info(f"[{i+1}/{len(params_to_test)}] {param_name}")

        if param_name not in combined_delta:
            logger.warning(f"  No delta found for param, skipping")
            skipped_count += 1
            continue

        try:
            result = strategy.test_param(param_name, harmful_prompts)
            results.append(result)

            if result.ablation_validated:
                if enable_multi_probe and result.weight_classification:
                    wc = result.weight_classification
                    status = f"VALIDATED ({wc.classification}, conf={wc.confidence:.2f})"
                else:
                    status = f"VALIDATED (harm={result.harmful_count}/{result.total_count})"
                validated_count += 1
            elif not result.coherent:
                status = "skipped (broke generation)"
                skipped_count += 1
            else:
                status = f"rejected (harm={result.harmful_count}/{result.total_count})"
                rejected_count += 1

            logger.info(f"  -> {status}")

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  OOM, skipping")
            gc.collect()
            torch.cuda.empty_cache()
            skipped_count += 1
        except Exception as e:
            logger.warning(f"  Error: {e}, skipping")
            skipped_count += 1

    # Cleanup
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build weight classifications dict
    weight_classifications = {}
    for result in results:
        if result.weight_classification:
            weight_classifications[result.param_name] = result.weight_classification

    return AblationResults(
        total_tested=len(params_to_test),
        validated_count=validated_count,
        rejected_count=rejected_count,
        skipped_count=skipped_count,
        param_results=results,
        weight_classifications=weight_classifications,
    )


def run_baseline_validation(
    run_path: Path,
    delta_scale: float = 2.0,
    num_prompts: int = 5,
    test_known_pairs: bool = False,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run baseline validation to verify ablation setup.

    Uses BaselineValidator from dcaf.ablation.

    Args:
        run_path: Path to DCAF run directory
        delta_scale: Scale factor for safety delta
        num_prompts: Number of prompts to test
        test_known_pairs: Whether to test known breaking/safe pairs

    Returns:
        Baseline validation results or None if failed
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.ablation import ModelStateManager, AblationConfig
    from dcaf.ablation.baselines import BaselineValidator
    from dcaf.storage import DeltaStore

    # Load metadata to get model name
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_path = run_path / "checkpoints" / "base.pt"
    delta_path = _select_target_delta_path(run_path, delta_store)

    if not base_path.exists() or delta_path is None:
        logger.error(
            f"Missing base checkpoint or target delta. "
            f"base={base_path.exists()}, delta={delta_path is not None}"
        )
        return None

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Device: {device}")

    # Load model from metadata
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        metadata.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    safety_delta = torch.load(delta_path, map_location="cpu", weights_only=True)

    config = AblationConfig(
        delta_scale=delta_scale,
        max_tokens=100,
        num_prompts=num_prompts,
    )

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta,
        delta_scale=delta_scale,
        device=device,
    )

    validator = BaselineValidator(model, tokenizer, state_manager, config)
    results = validator.validate_all(test_known_pairs=test_known_pairs)
    validator.print_summary(results)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "all_passed": results.all_passed,
        "base_model": {
            "passed": results.base_model.passed,
            "harm_rate": results.base_model.harm_rate,
        },
        "safe_model": {
            "passed": results.safe_model.passed,
            "harm_rate": results.safe_model.harm_rate,
        },
        "known_pairs": [
            {
                "name": r.name,
                "passed": r.passed,
                "expected_harmful": r.expected_harmful,
                "actual_harmful": r.actual_harmful,
            }
            for r in results.known_pairs
        ],
    }


def run_pair_ablation(
    run_path: Path,
    matching_params: List[str],
    top_k: int = 50,
    delta_scale: float = 2.0,
    num_prompts: int = 5,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run pair ablation testing.

    Uses PairAblation strategy from dcaf.ablation.strategies.

    Args:
        run_path: Path to DCAF run directory
        matching_params: Parameters to test
        top_k: Limit to top K parameters
        delta_scale: Scale factor for safety delta
        num_prompts: Number of prompts to test

    Returns:
        Pair ablation results or None if failed
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.ablation import ModelStateManager, AblationConfig
    from dcaf.ablation.baselines import COMPLETION_PROMPTS_VIOLENCE
    from dcaf.ablation.strategies.pair_ablation import PairAblation
    from dcaf.storage import DeltaStore

    # Load metadata to get model name
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_path = run_path / "checkpoints" / "base.pt"
    delta_path = _select_target_delta_path(run_path, delta_store)

    if not base_path.exists() or delta_path is None:
        logger.error("Missing base checkpoint or target delta")
        return None

    logger.info("\n" + "=" * 70)
    logger.info("PAIR ABLATION TESTING")
    logger.info("=" * 70)
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Device: {device}")

    # Load model from metadata
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        metadata.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    safety_delta = torch.load(delta_path, map_location="cpu", weights_only=True)

    config = AblationConfig(
        delta_scale=delta_scale,
        max_tokens=100,
        num_prompts=num_prompts,
    )

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta,
        delta_scale=delta_scale,
        device=device,
    )

    strategy = PairAblation(model, tokenizer, state_manager, config)
    results = strategy.run(
        matching_params[:top_k],
        COMPLETION_PROMPTS_VIOLENCE,
    )

    logger.info(f"Tested {len(results.all_results)} pairs")
    logger.info(f"Breaking pairs: {len(results.breaking_pairs)} ({results.break_rate:.1%})")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "total_pairs": len(results.all_results),
        "breaking_pairs": len(results.breaking_pairs),
        "break_rate": results.break_rate,
        "pairs": [
            {
                "param1": r.param1,
                "param2": r.param2,
                "harmful_count": r.harmful_count,
                "total_count": r.total_count,
                "is_breaking": r.breaks_safety,
            }
            for r in results.all_results
        ],
    }


def run_binary_ablation(
    run_path: Path,
    matching_params: List[str],
    top_k: int = 50,
    delta_scale: float = 2.0,
    num_prompts: int = 5,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run binary search ablation to find minimal critical parameter set.

    Uses BinarySearchAblation strategy from dcaf.ablation.strategies.

    Args:
        run_path: Path to DCAF run directory
        matching_params: Parameters to test
        top_k: Limit to top K parameters
        delta_scale: Scale factor for safety delta
        num_prompts: Number of prompts to test

    Returns:
        Binary search ablation results or None if failed
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.ablation import ModelStateManager, AblationConfig
    from dcaf.ablation.baselines import COMPLETION_PROMPTS_VIOLENCE
    from dcaf.ablation.strategies.binary_search import BinarySearchAblation
    from dcaf.storage import DeltaStore

    # Load metadata to get model name
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_path = run_path / "checkpoints" / "base.pt"
    delta_path = _select_target_delta_path(run_path, delta_store)

    if not base_path.exists() or delta_path is None:
        logger.error("Missing base checkpoint or target delta")
        return None

    logger.info("\n" + "=" * 70)
    logger.info("BINARY SEARCH ABLATION")
    logger.info("=" * 70)
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Device: {device}")

    # Load model from metadata
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        metadata.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    safety_delta = torch.load(delta_path, map_location="cpu", weights_only=True)

    config = AblationConfig(
        delta_scale=delta_scale,
        max_tokens=100,
        num_prompts=num_prompts,
    )

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta,
        delta_scale=delta_scale,
        device=device,
    )

    strategy = BinarySearchAblation(model, tokenizer, state_manager, config)
    results = strategy.run(
        matching_params[:top_k],
        COMPLETION_PROMPTS_VIOLENCE,
    )

    logger.info(f"Initial params: {len(results.initial_params)}")
    logger.info(f"Critical params: {len(results.critical_params)}")
    logger.info(f"Reduction: {results.reduction_ratio:.1%}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "initial_count": len(results.initial_params),
        "critical_count": len(results.critical_params),
        "reduction_ratio": results.reduction_ratio,
        "critical_params": results.critical_params,
        "iterations": results.iterations,
        "search_log": results.search_log,
    }


def run_group_ablation(
    run_path: Path,
    matching_params: List[str],
    top_k: int = 50,
    delta_scale: float = 2.0,
    num_prompts: int = 5,
    classifier_model: Optional[str] = None,
    classifier_8bit: bool = True,
    classifier_4bit: bool = False,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run group ablation testing on all params together.

    Uses GroupAblation strategy from dcaf.ablation.strategies.

    Args:
        run_path: Path to DCAF run directory
        matching_params: Parameters to test together
        top_k: Limit to top K parameters
        delta_scale: Scale factor for safety delta
        num_prompts: Number of prompts to test
        classifier_model: Override default classifier model
        classifier_8bit: Load classifier in 8-bit quantization
        classifier_4bit: Load classifier in 4-bit quantization

    Returns:
        Group ablation results or None if failed
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.ablation import ModelStateManager, AblationConfig
    from dcaf.ablation.strategies.group_ablation import GroupAblation
    from dcaf.evaluation.refusal import RefusalClassifier
    from dcaf.storage.delta_store import DeltaStore

    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_path = run_path / "checkpoints" / "base.pt"
    delta_path = _select_target_delta_path(run_path, delta_store)

    if not base_path.exists() or delta_path is None:
        logger.error("Missing base checkpoint or target delta")
        return None

    logger.info("\n" + "=" * 70)
    logger.info("GROUP ABLATION TESTING")
    logger.info("=" * 70)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        metadata.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    safety_delta = torch.load(delta_path, map_location="cpu", weights_only=True)

    # Load classifier
    try:
        classifier = RefusalClassifier(
            method="llm",
            device=device,
            model_name=classifier_model,
            load_in_8bit=classifier_8bit,
            load_in_4bit=classifier_4bit,
        )
    except Exception as e:
        logger.warning(f"Classifier failed ({e}), using heuristic")
        classifier = None

    config = AblationConfig(
        delta_scale=delta_scale,
        max_tokens=100,
        num_prompts=num_prompts,
    )

    state_manager = ModelStateManager(
        model, base_checkpoint, safety_delta,
        delta_scale=delta_scale,
        device=device,
    )

    strategy = GroupAblation(
        model, tokenizer, state_manager, config,
        classifier=classifier.classify if classifier else None,
    )

    # Test prompts
    harmful_prompts = REFUSAL_TEST_PROMPTS[:num_prompts]
    params_to_test = matching_params[:top_k]

    logger.info(f"Testing {len(params_to_test)} params as a group")

    result = strategy.run(params_to_test, harmful_prompts)

    logger.info(f"Safety broken: {result.safety_broken}")
    logger.info(f"Refusal rate: {result.refusal_rate:.1%}")

    # Cleanup
    if classifier:
        del classifier
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.to_dict()


__all__ = [
    "run_ablation_testing",
    "run_baseline_validation",
    "run_pair_ablation",
    "run_binary_ablation",
    "run_group_ablation",
]
