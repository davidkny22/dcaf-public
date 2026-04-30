#!/usr/bin/env python3
"""
DCAF Training CLI (§12 Complete Pipeline: training phase).

Train once on GPU; save deltas and checkpoints for later analysis.
Uses ``dcaf.training.variants.build_variant()`` to compose signal runs from
boolean flags (prefopt, sft, cumulative, anti, negated).

Usage:
    dcaf train --category Violence -o ./runs/violence_001/
    dcaf train --anti --category Violence -o ./runs/violence_anti/
    dcaf train --target-only --anti --category Violence -o ./runs/target_only/

Output directory layout:
    metadata.json     — run configuration and available deltas
    deltas/*.pt       — weight delta tensors (one per signal)
    checkpoints/*.pt  — checkpoints saved at training peaks
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import torch

from dcaf.cli.common import (
    detect_device,
    configure_logging,
    add_device_args,
    add_significance_threshold_args,
    add_probe_args,
    add_verbose_args,
)

# Centralized prompt bank
from dcaf.data.test_banks import REFUSAL_TEST_BANK
from dcaf.core.defaults import (
    SIMPO_BATCH_SIZE, SIMPO_GRAD_ACCUM,
    SFT_BATCH_SIZE, SFT_GRAD_ACCUM, SIMPO_BETA,
)

# Hardware-specific training presets
TRAINING_PRESETS = {
    "A100-S": {
        "simpo_batch_size": 2,
        "simpo_grad_accum": 32,
        "sft_batch_size": 4,
        "sft_grad_accum": 8,
        "beta": SIMPO_BETA,
    },
    "A100-L": {
        "simpo_batch_size": 4,
        "simpo_grad_accum": 16,
        "sft_batch_size": 8,
        "sft_grad_accum": 4,
        "beta": SIMPO_BETA,
    },
}

logger = logging.getLogger(__name__)


# Known categories and groups (from demo_mvp.py)
KNOWN_CATEGORIES = [
    "Violence", "Physical Harm", "Psychological Harm", "Sexual Content",
    "Drugs", "Cybercrime", "Privacy Violation", "Economic Crime",
    "White-Collar Crime", "Discriminatory Behavior", "Insulting Behavior",
    "Mental Manipulation", "Human Trafficking", "Animal Abuse",
    "Endangering National Security", "Endangering Public Health",
    "Disrupting Public Order", "Environmental Damage", "Copyright Issues",
]

KNOWN_GROUPS = [
    "violence", "psychological", "illegal", "discrimination",
    "privacy", "harmful_content", "public_safety",
]


def _require_non_empty(name: str, data, args) -> None:
    """Fail early with filter context when a required dataset is empty."""
    try:
        size = len(data)
    except TypeError:
        return
    if size == 0:
        filters = {
            "category": args.category,
            "category_group": args.category_group,
            "min_severity": args.min_severity,
            "samples": args.samples,
        }
        raise ValueError(f"{name} is empty for filters {filters}")


def create_argparser() -> argparse.ArgumentParser:
    """Create argparser with training-relevant flags."""
    parser = argparse.ArgumentParser(
        description="DCAF Training Phase (GPU) - Save deltas for later analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base: PO (both sides) + language baseline
  python dcaf-train --category Violence -o ./runs/violence_001/

  # Add Anti-training modifier (both sides)
  python dcaf-train --anti --category Violence -o ./runs/violence_anti/

  # All modifiers
  python dcaf-train --sft --cumulative --anti --negated --category Violence -o ./runs/violence_scan/

  # Target-only (T+ side only)
  python dcaf-train --target-only --anti --category Violence -o ./runs/violence_target/

  # Category intersection mode
  python dcaf-train --category-intersection -o ./runs/intersection_001/
""",
    )

    # Required
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for deltas and metadata",
    )

    # Model
    parser.add_argument(
        "--model", default="EleutherAI/pythia-410m",
        help="Model to use (default: EleutherAI/pythia-410m)",
    )

    # UnSloth options
    parser.add_argument(
        "--no-unsloth", action="store_true",
        help="Disable UnSloth and use standard transformers (slower, for debugging)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Maximum sequence length for UnSloth (default: 2048)",
    )

    # Modifier flags
    parser.add_argument(
        "--anti", "-a", action="store_true",
        help="Add Anti-training runs (both sides, default: False)",
    )
    parser.add_argument(
        "--negated", "-n", action="store_true",
        help="Add NegatedSimPO runs (both sides, default: False)",
    )
    parser.add_argument(
        "--sft", "-f", action="store_true",
        help="Add SFT training runs (both sides, default: False)",
    )
    parser.add_argument(
        "--cumulative", "-c", action="store_true",
        help="Add cumulative runs (SFT->SimPO sequential, both sides, default: False)",
    )

    # Direction control
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--target-only", action="store_true",
        help="Only run T+ (target-side) training. Default is both sides.",
    )
    direction_group.add_argument(
        "--opposite-only", action="store_true",
        help="Only run T- (opposite-side) training. Default is both sides.",
    )

    # Dataset options
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Limit number of SafeRLHF samples (default: all available)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by single harm category",
    )
    parser.add_argument(
        "--category-group", type=str, default=None,
        help="Filter by category group (violence, psychological, illegal, etc.)",
    )
    parser.add_argument(
        "--min-severity", type=int, default=0, choices=[0, 1, 2, 3],
        help="Minimum severity level for unsafe responses (default: 0=all)",
    )

    # Training options
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(TRAINING_PRESETS.keys()),
        help="Training preset: A100-S (<5k samples), A100-L (>5k samples)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs per phase (default: 1, full pass through dataset)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="Override epochs with fixed step count (default: -1 = use epochs)",
    )
    parser.add_argument(
        "--no-simpo", action="store_true",
        help="Use SFT-only training (no preference optimization). Replaces PrefOpt signals with SFT equivalents.",
    )
    parser.add_argument(
        "--simpo-batch-size", type=int, default=None,
        help="SimPO batch size (overrides preset, default: 2)",
    )
    parser.add_argument(
        "--simpo-grad-accum", type=int, default=None,
        help="SimPO gradient accumulation steps (overrides preset, default: 32)",
    )
    parser.add_argument(
        "--sft-batch-size", type=int, default=None,
        help="SFT batch size (overrides preset, default: 2)",
    )
    parser.add_argument(
        "--sft-grad-accum", type=int, default=None,
        help="SFT gradient accumulation steps (overrides preset, default: 8)",
    )
    parser.add_argument(
        "--simpo-beta", type=float, default=None,
        help="SimPO beta parameter (overrides preset, default: 5.0)",
    )
    add_significance_threshold_args(parser)

    # Category intersection
    parser.add_argument(
        "--category-intersection", action="store_true",
        help="Run DCAF for each category and find intersection (params common to all)",
    )
    parser.add_argument(
        "--intersection-categories", type=str, default=None,
        help="Comma-separated categories for intersection (default: Violence,Cybercrime,Privacy Violation)",
    )

    # Refusal tracking
    parser.add_argument(
        "--track-refusal", action="store_true",
        help="Measure refusal rate after each training phase (saves to metadata)",
    )

    # Device
    add_device_args(parser)

    # Activation delta capture (captures after EACH training run)
    parser.add_argument(
        "--no-capture-activations", action="store_true",
        help="Disable activation capture after each training run (enabled by default)",
    )
    add_probe_args(parser, default=100)
    # Hidden alias kept so old scripts don't break
    parser.add_argument(
        "--probe-set-size", type=int, dest="probe_size",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--activation-probe-type", type=str, default="both",
        choices=["recognition", "teacher_forcing", "both"],
        help="Type of probes to capture: recognition (harmful+neutral forward), "
             "teacher_forcing (safe/unsafe prefix), or both (default: both)",
    )
    parser.add_argument(
        "--enable-free-generation", action="store_true",
        help="Enable free generation probe (steering decision, 10 tokens, slow) (default: False)",
    )
    parser.add_argument(
        "--free-gen-max-tokens", type=int, default=10,
        help="Max tokens for free generation steering window (default: 10)",
    )
    parser.add_argument(
        "--activation-batch-size", type=int, default=8,
        help="Batch size for activation capture (default: 8)",
    )

    add_verbose_args(parser)

    return parser


def build_modifiers_string(args) -> str:
    """Build modifier string from CLI flags."""
    mods = ""
    if args.sft:
        mods += "S"
    if args.cumulative:
        mods += "C"
    if args.anti:
        mods += "A"
    if args.negated:
        mods += "N"
    return mods


def run_train(args):
    """Run training with pre-parsed arguments."""
    # Build variant from flags
    from dcaf.training.variants import build_variant

    modifiers = build_modifiers_string(args)
    target = not args.opposite_only
    opposite = not args.target_only
    variant_config = build_variant(
        modifiers, target=target, opposite=opposite, no_simpo=args.no_simpo,
    )

    # Apply training preset if specified
    if args.preset:
        preset = TRAINING_PRESETS[args.preset]
        # Only override if not explicitly set by user
        if args.simpo_batch_size is None:
            args.simpo_batch_size = preset["simpo_batch_size"]
        if args.simpo_grad_accum is None:
            args.simpo_grad_accum = preset["simpo_grad_accum"]
        if args.sft_batch_size is None:
            args.sft_batch_size = preset["sft_batch_size"]
        if args.sft_grad_accum is None:
            args.sft_grad_accum = preset["sft_grad_accum"]
        if args.simpo_beta is None:
            args.simpo_beta = preset["beta"]
    else:
        # No preset: use defaults from defaults.py
        if args.simpo_batch_size is None:
            args.simpo_batch_size = SIMPO_BATCH_SIZE
        if args.simpo_grad_accum is None:
            args.simpo_grad_accum = SIMPO_GRAD_ACCUM
        if args.sft_batch_size is None:
            args.sft_batch_size = SFT_BATCH_SIZE
        if args.sft_grad_accum is None:
            args.sft_grad_accum = SFT_GRAD_ACCUM
        if args.simpo_beta is None:
            args.simpo_beta = SIMPO_BETA

    # Validate category
    if args.category:
        matched = next(
            (c for c in KNOWN_CATEGORIES if c.lower() == args.category.lower()),
            None,
        )
        if not matched:
            logger.error(f"Unknown category: {args.category}")
            logger.error(f"Available: {', '.join(KNOWN_CATEGORIES)}")
            sys.exit(1)
        args.category = matched

    # Validate category group
    if args.category_group:
        if args.category_group.lower() not in KNOWN_GROUPS:
            logger.error(f"Unknown category group: {args.category_group}")
            logger.error(f"Available: {', '.join(KNOWN_GROUPS)}")
            sys.exit(1)
        args.category_group = args.category_group.lower()

    # Determine device
    device = detect_device(args.device)
    if device == "cpu":
        logger.warning("No GPU available. Training on CPU will be slow.")

    variant_name = variant_config.name

    logger.info("=" * 70)
    logger.info("DCAF Training Phase")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Variant: {variant_name}")
    logger.info(f"  Runs: {len(variant_config.runs)}")
    for run in variant_config.runs:
        logger.info(f"    - {run.run_type} (delta={run.delta_name})")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {device}")
    if args.max_steps > 0:
        logger.info(f"Max steps per phase: {args.max_steps}")
    else:
        logger.info(f"Epochs per phase: {args.epochs}")
    logger.info(f"Samples: {args.samples}")
    if args.preset:
        logger.info(f"Training preset: {args.preset}")
    logger.info(f"SimPO: batch={args.simpo_batch_size}, grad_accum={args.simpo_grad_accum}, beta={args.simpo_beta} (effective={args.simpo_batch_size * args.simpo_grad_accum})")
    logger.info(f"SFT: batch={args.sft_batch_size}, grad_accum={args.sft_grad_accum} (effective={args.sft_batch_size * args.sft_grad_accum})")
    if args.category:
        logger.info(f"Category filter: {args.category}")
    if args.category_group:
        logger.info(f"Category group: {args.category_group}")
    if args.min_severity > 0:
        logger.info(f"Min severity: {args.min_severity}")
    if args.category_intersection:
        cats = args.intersection_categories or "Violence,Cybercrime,Privacy Violation"
        logger.info(f"Category intersection: {cats}")
    logger.info("=" * 70)

    # Import heavy modules after argument parsing
    from dcaf.arch.model_loading import load_model_for_training
    from dcaf.storage.delta_store import DeltaStore, DeltaMetadata
    from dcaf.training.variants import TrainingOrchestrator
    from dcaf.core.config import DCAFConfig
    from dcaf.data.safe_rlhf import SafeRLHFLoader

    # Load model and tokenizer (UnSloth by default for faster training)
    logger.info(f"\nLoading model: {args.model}")
    use_unsloth = not args.no_unsloth
    model, tokenizer = load_model_for_training(
        model_name=args.model,
        use_unsloth=use_unsloth,
        device=device,
        max_seq_length=args.max_seq_length,
    )

    # Load datasets
    logger.info("\nLoading SafeRLHF datasets...")
    loader = SafeRLHFLoader(
        tokenizer=tokenizer,
        max_length=256,
        subset_size=args.samples,
    )

    # Apply category filter
    category_filter = None
    if args.category:
        category_filter = [args.category]
    elif args.category_group:
        category_filter = args.category_group

    # Create datasets
    datasets = {}

    # SimPO datasets
    if not args.no_simpo:
        logger.info("  Creating SimPO safe dataset...")
        datasets["safe_simpo"] = loader.create_simpo_dataset(
            categories=category_filter,
            min_severity=args.min_severity,
        )
        _require_non_empty("safe_simpo", datasets["safe_simpo"], args)
        logger.info(f"    {len(datasets['safe_simpo'])} samples")

        # Adversarial dataset is needed for T- runs and for target-side
        # anti-training, which ascends on the opposite/unsafe preference set.
        if opposite or (target and args.anti):
            logger.info("  Creating SimPO adversarial dataset...")
            datasets["unsafe_simpo"] = loader.create_adversarial_simpo_dataset(
                categories=category_filter,
                min_severity=args.min_severity,
            )
            _require_non_empty("unsafe_simpo", datasets["unsafe_simpo"], args)
            logger.info(f"    {len(datasets['unsafe_simpo'])} samples")

    # SFT dataloaders for SFT-backed runs.
    needs_sft = args.no_simpo or args.sft or args.cumulative
    if needs_sft:
        from torch.utils.data import DataLoader
        logger.info("  Creating SFT dataloaders...")

        # Create unsafe dataset with category filter
        unsafe_sft_dataset = loader.create_sft_unsafe_dataset(
            categories=category_filter,
            min_severity=args.min_severity,
        )
        _require_non_empty("unsafe_sft_dataset", unsafe_sft_dataset, args)

        # Create safe dataloader with collate_fn for tokenization
        safe_sft_dataset = loader.create_sft_safe_dataset(
            categories=category_filter,
            min_severity=args.min_severity,
        )
        _require_non_empty("safe_sft_dataset", safe_sft_dataset, args)

        def collate_fn(examples):
            texts = [ex["text"] for ex in examples]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        datasets["safe_sft_dataloader"] = DataLoader(
            safe_sft_dataset,
            batch_size=args.sft_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        datasets["unsafe_sft_dataloader"] = DataLoader(
            unsafe_sft_dataset,
            batch_size=args.sft_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        logger.info(f"    Safe batches: {len(datasets['safe_sft_dataloader'])}")
        logger.info(f"    Unsafe batches: {len(datasets['unsafe_sft_dataloader'])}")

    # Language baseline (always needed - build_variant always includes it)
    from dcaf.data import create_neutral_dataloader
    logger.info("  Creating language baseline dataloader...")
    datasets["language_dataloader"] = create_neutral_dataloader(
        tokenizer=tokenizer,
        batch_size=args.sft_batch_size,
    )
    logger.info(f"    Batches: {len(datasets['language_dataloader'])}")

    # Create output directory and DeltaStore
    output_path = Path(args.output)
    delta_store = DeltaStore(output_path)

    # Create metadata
    metadata = DeltaMetadata.create(
        model_name=args.model,
        variant_name=variant_name,
        training_config={
            "epochs_per_phase": args.epochs,
            "max_steps_per_phase": args.max_steps,
            "no_simpo": args.no_simpo,
            "percentile": args.tau_sig,
            "language_percentile": args.language_percentile,
            "device": device,
            "use_unsloth": not args.no_unsloth,
            "max_seq_length": args.max_seq_length,
        },
        dataset_config={
            "samples": args.samples,
            "category": args.category,
            "category_group": args.category_group,
            "min_severity": args.min_severity,
        },
    )
    delta_store.save_metadata(metadata)

    # Create DCAF config and orchestrator
    config = DCAFConfig(
        use_simpo=not args.no_simpo,
        simpo_beta=args.simpo_beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.simpo_batch_size,
        gradient_accumulation_steps=args.simpo_grad_accum,
        sft_batch_size=args.sft_batch_size,
        sft_gradient_accumulation_steps=args.sft_grad_accum,
    )
    orchestrator = TrainingOrchestrator(model, tokenizer, config, device)

    # Enable activation delta capture (on by default, --no-capture-activations to disable)
    if not args.no_capture_activations:
        logger.info("\nEnabling activation delta capture...")

        # Create probe set (matched contrast pairs when available)
        from dcaf.domains.activation import ProbeSet
        category_lower = args.category.lower() if args.category else None
        probe_set_source = "matched_pairs"
        try:
            probe_set = ProbeSet.from_pairs(
                category=category_lower,
                size=args.probe_size,
                seed=42,
            )
            logger.info(f"Using matched contrast pairs: {len(probe_set)} prompts")
        except Exception as exc:
            probe_set_source = "default_fallback"
            logger.warning(f"Matched-pair probe set unavailable ({exc}); using default probes")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                probe_set = ProbeSet.default(size=args.probe_size, seed=42)

        # Save probe set to delta store
        delta_store.save_probe_set(probe_set)

        # Enable capture on orchestrator
        orchestrator.enable_activation_capture(
            probe_set=probe_set,
            delta_store=delta_store,
            probe_type=args.activation_probe_type,
            batch_size=args.activation_batch_size,
            enable_free_generation=args.enable_free_generation,
            free_gen_max_tokens=args.free_gen_max_tokens,
        )
        orchestrator._activation_config["probe_set_source"] = probe_set_source

        # Capture base activations BEFORE training
        logger.info("Capturing pre-training baseline activations...")
        base_snapshot = orchestrator._activation_capturer.capture(
            probe_set=probe_set,
            tokenizer=tokenizer,
            name="pre_training",
            probe_type=args.activation_probe_type,
            max_length=128,
            batch_size=args.activation_batch_size,
            enable_free_generation=args.enable_free_generation,
            max_new_tokens=args.free_gen_max_tokens,
        )
        delta_store.save_activation_snapshot("pre_training", base_snapshot)
        logger.info("  Pre-training baseline saved")

        # Update metadata
        metadata.activation_capture_enabled = True
        metadata.probe_set_name = probe_set.name
        metadata.probe_set_size = len(probe_set)
        metadata.activation_config = orchestrator._activation_config
        delta_store.save_metadata(metadata)

    # Category intersection mode
    if args.category_intersection:
        # Parse categories
        if args.intersection_categories:
            categories = [c.strip() for c in args.intersection_categories.split(",")]
        else:
            categories = ["Violence", "Cybercrime", "Privacy Violation"]

        # Validate categories
        valid_categories = []
        for cat in categories:
            matched = next(
                (c for c in KNOWN_CATEGORIES if c.lower() == cat.lower()), None
            )
            if matched:
                valid_categories.append(matched)
            else:
                logger.warning(f"Unknown category '{cat}', skipping")

        if len(valid_categories) < 2:
            logger.error("Need at least 2 valid categories for intersection")
            sys.exit(1)

        logger.info(f"\n{'=' * 70}")
        logger.info("CATEGORY INTERSECTION MODE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Running {variant_name} for {len(valid_categories)} categories:")
        for cat in valid_categories:
            logger.info(f"  - {cat}")

        per_category_params = {}

        for i, category in enumerate(valid_categories):
            logger.info(f"\n[{i+1}/{len(valid_categories)}] Running DCAF for: {category}")
            logger.info("-" * 50)

            # Create category-specific output directory
            cat_output = output_path / f"category_{category.replace(' ', '_')}"
            cat_delta_store = DeltaStore(cat_output)

            # Create category-specific metadata
            cat_metadata = DeltaMetadata.create(
                model_name=args.model,
                variant_name=variant_name,
                training_config={
                    "epochs_per_phase": args.epochs,
                    "max_steps_per_phase": args.max_steps,
                    "no_simpo": args.no_simpo,
                    "percentile": args.tau_sig,
                    "language_percentile": args.language_percentile,
                    "device": device,
                    "intersection_category": category,
                },
                dataset_config={
                    "samples": args.samples,
                    "category": category,
                    "min_severity": args.min_severity,
                },
            )
            if not args.no_capture_activations:
                cat_metadata.activation_capture_enabled = True
                cat_metadata.probe_set_name = probe_set.name
                cat_metadata.probe_set_size = len(probe_set)
                cat_metadata.activation_config = orchestrator._activation_config
            cat_delta_store.save_metadata(cat_metadata)

            cat_simpo = None
            cat_unsafe_simpo = None
            cat_safe_sft_dataloader = datasets.get("safe_sft_dataloader")
            cat_unsafe_sft_dataloader = datasets.get("unsafe_sft_dataloader")

            if args.no_simpo:
                cat_safe_sft_dataset = loader.create_sft_safe_dataset(
                    categories=[category],
                    min_severity=args.min_severity,
                )
                cat_unsafe_sft_dataset = loader.create_sft_unsafe_dataset(
                    categories=[category],
                    min_severity=args.min_severity,
                )

                if len(cat_safe_sft_dataset) < 10:
                    logger.warning(
                        f"Only {len(cat_safe_sft_dataset)} SFT samples for {category}, skipping"
                    )
                    continue

                cat_safe_sft_dataloader = DataLoader(
                    cat_safe_sft_dataset,
                    batch_size=args.sft_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                cat_unsafe_sft_dataloader = DataLoader(
                    cat_unsafe_sft_dataset,
                    batch_size=args.sft_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                logger.info(f"  SFT dataset size: {len(cat_safe_sft_dataset)} safe examples")
                logger.info(f"  SFT adversarial size: {len(cat_unsafe_sft_dataset)} unsafe examples")
            else:
                # Create category-specific SimPO dataset
                cat_simpo = loader.create_simpo_dataset(
                    categories=[category],
                    min_severity=args.min_severity,
                )

                if len(cat_simpo) < 10:
                    logger.warning(f"Only {len(cat_simpo)} samples for {category}, skipping")
                    continue

                logger.info(f"  Dataset size: {len(cat_simpo)} preference pairs")

                # Create adversarial data for T- runs and target-side anti-training.
                if not args.target_only or args.anti:
                    cat_unsafe_simpo = loader.create_adversarial_simpo_dataset(
                        categories=[category],
                        min_severity=args.min_severity,
                    )
                    logger.info(f"  Adversarial size: {len(cat_unsafe_simpo)} pairs")

            # Run DCAF for this category (training only, no analysis)
            try:
                orchestrator.run_variant(
                    variant_config=variant_config,
                    safe_simpo_dataset=cat_simpo,
                    unsafe_simpo_dataset=cat_unsafe_simpo,
                    safe_sft_dataloader=cat_safe_sft_dataloader,
                    unsafe_sft_dataloader=cat_unsafe_sft_dataloader,
                    language_dataloader=datasets.get("language_dataloader"),
                    epochs_per_phase=args.epochs,
                    max_steps_per_phase=args.max_steps,
                    delta_store=cat_delta_store,
                )

                # Track which categories completed successfully
                per_category_params[category] = cat_output
                logger.info(f"  Saved deltas to: {cat_output}")

                # Update category metadata
                cat_metadata.available_deltas = cat_delta_store.list_deltas()
                cat_metadata.available_checkpoints = cat_delta_store.list_checkpoints()
                cat_delta_store.save_metadata(cat_metadata)

            except Exception as e:
                logger.error(f"  DCAF failed for {category}: {e}")
                continue

            # Clear CUDA cache between runs
            if device == "cuda":
                torch.cuda.empty_cache()

        # Save category run info (analysis happens in dcaf-analyze --cross-run)
        category_runs = {
            "type": "category_intersection_training",
            "categories": list(per_category_params.keys()),
            "run_directories": {k: str(v) for k, v in per_category_params.items()},
        }
        import json
        with open(output_path / "category_runs.json", "w") as f:
            json.dump(category_runs, f, indent=2)

        # Update main metadata
        metadata.training_config["category_runs"] = list(per_category_params.keys())
        delta_store.save_metadata(metadata)

        logger.info(f"\n{'=' * 70}")
        logger.info("CATEGORY INTERSECTION TRAINING COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Categories trained: {len(per_category_params)}")
        for cat, path in per_category_params.items():
            logger.info(f"  {cat}: {path}")
        logger.info(f"\nRun directories saved to: {output_path / 'category_runs.json'}")
        logger.info(f"\nNext step: Analyze intersection with dcaf-analyze")
        logger.info(f"  python dcaf-analyze --cross-run {' '.join(str(p) for p in per_category_params.values())}")

    else:
        # Normal mode: Run variant in save mode
        logger.info(f"\nRunning {variant_name}...")

        # Create refusal rate callback if tracking enabled
        phase_callback = None
        if args.track_refusal:
            logger.info("Refusal tracking enabled - will measure after each phase")

            # Select prompts based on training category or category group
            from dcaf.data.safe_rlhf import HARM_CATEGORY_GROUPS

            if args.category and args.category in REFUSAL_TEST_BANK:
                # Use category-specific prompts
                refusal_test_prompts = REFUSAL_TEST_BANK[args.category]
                logger.info(f"Using {len(refusal_test_prompts)} refusal prompts for category: {args.category}")
            elif args.category_group and args.category_group in HARM_CATEGORY_GROUPS:
                # Use prompts from all categories in the group
                refusal_test_prompts = []
                group_categories = HARM_CATEGORY_GROUPS[args.category_group]
                for cat in group_categories:
                    if cat in REFUSAL_TEST_BANK:
                        refusal_test_prompts.extend(REFUSAL_TEST_BANK[cat])
                logger.info(f"Using {len(refusal_test_prompts)} refusal prompts for category group '{args.category_group}': {', '.join(group_categories)}")
            else:
                # Use 25 prompts across categories for unspecified runs
                refusal_test_prompts = []
                categories_to_sample = ["Violence", "Cybercrime", "Drugs", "Privacy Violation", "Mental Manipulation"]
                for cat in categories_to_sample:
                    refusal_test_prompts.extend(REFUSAL_TEST_BANK[cat][:5])
                logger.info(f"Using {len(refusal_test_prompts)} refusal prompts (5 per category, 5 categories)")

            # Initialize refusal classifier
            from dcaf.evaluation.refusal import RefusalClassifier
            logger.info("Loading refusal classifier...")
            refusal_classifier = RefusalClassifier(
                method="llm",
                device=device,
                load_in_8bit=True,  # Save VRAM
            )

            def measure_refusal_rate(phase_name: str, phase_idx: int) -> dict:
                """Measure refusal rate on test prompts using LLM classifier."""
                import json  # Import here to avoid scoping issues with nested functions
                model.eval()
                refusals = 0
                complies = 0
                irrelevant = 0

                # Store all prompts, responses, and classifications for post-hoc verification
                detailed_results = []

                for prompt in refusal_test_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=False,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    prompt_len = inputs["input_ids"].shape[-1]
                    response = tokenizer.decode(
                        outputs[0][prompt_len:],
                        skip_special_tokens=True,
                    )

                    # Use classifier to determine if refusal
                    result = refusal_classifier.classify(response, prompt)

                    # Store detailed result
                    detailed_results.append({
                        "prompt": prompt,
                        "response": response,
                        "is_refusal": result.is_refusal,
                        "confidence": result.confidence,
                        "category": result.category.value if result.category else None,
                        "reasoning": result.reasoning
                    })

                    if result.is_refusal is None:
                        # Irrelevant response - exclude from counts
                        irrelevant += 1
                    elif result.is_refusal:
                        refusals += 1
                    else:
                        complies += 1

                total_relevant = refusals + complies
                rate = refusals / total_relevant if total_relevant > 0 else 0
                logger.info(f"  Refusal rate after {phase_name}: {rate:.1%} ({refusals}/{total_relevant}, {irrelevant} irrelevant)")

                # Save detailed results to JSON file for post-hoc verification
                refusal_results_dir = output_path / "refusal_tracking"
                refusal_results_dir.mkdir(exist_ok=True)
                refusal_results_file = refusal_results_dir / f"phase_{phase_idx:02d}_{phase_name.replace(' ', '_')}.json"

                with open(refusal_results_file, "w") as f:
                    json.dump({
                        "phase_name": phase_name,
                        "phase_idx": phase_idx,
                        "summary": {
                            "refusal_rate": rate,
                            "refusals": refusals,
                            "complies": complies,
                            "irrelevant": irrelevant,
                            "total": len(refusal_test_prompts)
                        },
                        "detailed_results": detailed_results
                    }, f, indent=2)

                logger.info(f"  Saved detailed refusal tracking to: {refusal_results_file}")

                return {
                    "refusal_rate": rate,
                    "refusals": refusals,
                    "complies": complies,
                    "irrelevant": irrelevant,
                    "total": len(refusal_test_prompts)
                }

            phase_callback = measure_refusal_rate

        orchestrator.run_variant(
            variant_config=variant_config,
            safe_simpo_dataset=datasets.get("safe_simpo"),
            unsafe_simpo_dataset=datasets.get("unsafe_simpo"),
            safe_sft_dataloader=datasets.get("safe_sft_dataloader"),
            unsafe_sft_dataloader=datasets.get("unsafe_sft_dataloader"),
            language_dataloader=datasets.get("language_dataloader"),
            epochs_per_phase=args.epochs,
            max_steps_per_phase=args.max_steps,
            delta_store=delta_store,
            phase_callback=phase_callback,
        )

        # Update metadata with available deltas (normal mode only)
        metadata.available_deltas = delta_store.list_deltas()
        metadata.available_checkpoints = delta_store.list_checkpoints()
        delta_store.save_metadata(metadata)

        # Summary (normal mode only)
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Output: {output_path}")
        logger.info(f"Deltas saved: {len(metadata.available_deltas)}")
        for d in metadata.available_deltas:
            logger.info(f"  - {d}")
        if metadata.available_checkpoints:
            logger.info(f"Checkpoints saved: {len(metadata.available_checkpoints)}")
            for c in metadata.available_checkpoints:
                logger.info(f"  - {c}")
        logger.info("")
        logger.info("Next step: Analyze with dcaf analyze")
        logger.info(f"  dcaf analyze -r {output_path}")


def main():
    """Standalone entry point for dcaf-train."""
    parser = create_argparser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    run_train(args)


__all__ = ["create_argparser", "run_train", "main"]


if __name__ == "__main__":
    main()
