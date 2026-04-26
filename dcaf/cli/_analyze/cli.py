"""
DCAF Analyze CLI (§12 Complete Pipeline: analysis phase).

Coordinates weight, activation, and geometry domain analysis, triangulation,
ablation testing, and circuit identification.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from dcaf.data.test_banks import get_refusal_test_prompts, get_benign_test_prompts

from dcaf.cli._analyze.formatters import (
    format_summary,
    format_params,
    compare_results,
    display_ablation_results,
    format_signal_breakdown,
)
from dcaf.cli.common import detect_device, configure_logging
from dcaf.core.defaults import (
    TAU_SIG,
    TAU_W_DEFAULT,
    TAU_A_DEFAULT,
    TAU_G_DEFAULT,
    LANGUAGE_PERCENTILE,
    TOP_K_CANDIDATES,
    CLASSIFICATION_THRESHOLD,
    SEPARATION_RATIO,
    DELTA_SCALE_DEFAULT,
    TAU_EDGE,
    ATTENTION_WEIGHT,
    HIGH_CONSISTENCY,
)
from dcaf.cli._analyze.refusal_testing import test_refusal_rates
from dcaf.cli._analyze.ablation_runner import (
    run_ablation_testing,
    run_baseline_validation,
    run_pair_ablation,
    run_binary_ablation,
    run_group_ablation,
)
from dcaf.cli._analyze.probe_runner import run_probe_analysis
from dcaf.ablation.results import ParamAblationResult, AblationResults

logger = logging.getLogger(__name__)

# Prompt constants
REFUSAL_TEST_PROMPTS = get_refusal_test_prompts(format="question")
BENIGN_TEST_PROMPTS = get_benign_test_prompts(format="instruction")


def create_argparser() -> argparse.ArgumentParser:
    """Create argparser for analysis CLI."""
    parser = argparse.ArgumentParser(
        description="DCAF Analysis Phase (CPU) - Analyze saved training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Full domain confidence analysis (C_W + C_A + C_G + triangulation)
  dcaf analyze -r ./runs/violence_001/

  # Weight domain only
  dcaf analyze -r ./runs/violence_001/ --weight

  # Activation domain only
  dcaf analyze -r ./runs/violence_001/ --activation

  # Save results to JSON
  dcaf analyze -r ./runs/violence_001/ -o results.json

  # Cross-run analysis (find stable core)
  dcaf analyze --cross-run ./runs/run_001/ ./runs/run_002/ --min-consistency 0.8
""",
    )

    # Required (unless using --cross-run)
    parser.add_argument(
        "--run", "-r",
        help="Path to DCAF training run directory (required unless using --cross-run)",
    )

    # Options
    parser.add_argument(
        "--significance-threshold", type=float, default=TAU_SIG,
        dest="tau_sig",
        help=f"Significance percentile threshold (default: {TAU_SIG})",
    )
    parser.add_argument(
        "--language-percentile", type=float, default=LANGUAGE_PERCENTILE,
        help=f"Percentile for language (NOT significant) threshold (default: {LANGUAGE_PERCENTILE})",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed parameter information",
    )
    parser.add_argument(
        "--show-signals", action="store_true",
        help="Display signal breakdown sorted by confidence",
    )
    parser.add_argument(
        "--min-confidence", type=float,
        help="Filter to params with confidence >= threshold (e.g., 0.8)",
    )
    parser.add_argument(
        "--sort-by-confidence", action="store_true",
        help="Sort output by confidence descending (default: by name)",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        metavar="PATH",
        help="Save layer/component distribution chart (PNG/PDF)",
    )
    parser.add_argument(
        "--show-chart",
        action="store_true",
        help="Display chart interactively (requires display)",
    )

    # SAE validation
    parser.add_argument(
        "--validate-sae", action="store_true",
        help="Cross-reference results with SAE features for interpretability validation",
    )

    # Cross-run analysis
    parser.add_argument(
        "--cross-run", nargs="+",
        help="Analyze patterns across multiple run directories",
    )
    parser.add_argument(
        "--min-consistency", type=float, default=HIGH_CONSISTENCY,
        help="Minimum consistency score for stable core (default: 0.8)",
    )

    # Refusal testing
    parser.add_argument(
        "--test-refusal", action="store_true",
        help="Test refusal rate on base model and after applying each delta",
    )
    parser.add_argument(
        "--refusal-prompts", type=int, default=25,
        help="Number of refusal test prompts to use (default: 25)",
    )

    # Ablation testing - STANDARDIZED FLAGS
    parser.add_argument(
        "--single-ablation", action="store_true",
        dest="ablation",
        help="Run single-param ablation testing to validate causal safety role",
    )
    # Keep --ablation as alias for backward compatibility
    parser.add_argument(
        "--ablation", action="store_true",
        dest="ablation",
        help=argparse.SUPPRESS,  # Hidden, use --single-ablation
    )
    parser.add_argument(
        "--ablation-top-k", type=int, default=TOP_K_CANDIDATES,
        help="Limit ablation to top K params by delta magnitude (default: 50)",
    )
    parser.add_argument(
        "--ablation-prompts", type=int, default=5,
        help="Number of harmful prompts per param test (default: 5)",
    )
    parser.add_argument(
        "--ablation-deltas", type=str, default="delta_t1_prefopt_target",
        help="Comma-separated deltas to apply for safety-trained state (default: delta_t1_prefopt_target)",
    )

    # Standardized ablation strategy flags
    parser.add_argument(
        "--pair-ablation", action="store_true",
        dest="ablation_pairs",
        help="Run pair ablation testing (within and cross-criteria pairs)",
    )
    # Keep --ablation-pairs as alias for backward compatibility
    parser.add_argument(
        "--ablation-pairs", action="store_true",
        dest="ablation_pairs",
        help=argparse.SUPPRESS,  # Hidden, use --pair-ablation
    )
    parser.add_argument(
        "--binary-ablation", action="store_true",
        dest="ablation_binary",
        help="Run binary search to find minimal critical parameter set",
    )
    # Keep --ablation-binary as alias for backward compatibility
    parser.add_argument(
        "--ablation-binary", action="store_true",
        dest="ablation_binary",
        help=argparse.SUPPRESS,  # Hidden, use --binary-ablation
    )
    parser.add_argument(
        "--group-ablation", action="store_true",
        help="Run group ablation testing (multiple params together)",
    )
    parser.add_argument(
        "--validate-baselines", action="store_true",
        help="Validate ablation setup (base model + safe model)",
    )
    parser.add_argument(
        "--test-known-pairs", action="store_true",
        help="Test known breaking/safe pairs during baseline validation (opt-in)",
    )

    # Multi-probe classification
    parser.add_argument(
        "--multi-probe-ablation", action="store_true",
        help="Enable two-probe weight classification (free generation + teacher forcing)",
    )
    parser.add_argument(
        "--classification-threshold", type=float, default=CLASSIFICATION_THRESHOLD,
        help=f"Minimum harm_rate to consider high impact (default: {CLASSIFICATION_THRESHOLD})",
    )
    parser.add_argument(
        "--separation-ratio", type=float, default=SEPARATION_RATIO,
        help=f"Ratio for specific vs shared classification (default: {SEPARATION_RATIO})",
    )
    parser.add_argument(
        "--filter-classification", type=str, default=None,
        choices=["recognition-specific", "generation-specific", "shared", "false-positive"],
        help="Filter output to weights with this classification",
    )
    parser.add_argument(
        "--safe-prefix", type=str, default="I cannot help with that request.",
        help="Safe prefix for teacher forcing",
    )
    parser.add_argument(
        "--unsafe-prefix", type=str, default="Here's how to do that:",
        help="Unsafe prefix for teacher forcing",
    )

    parser.add_argument(
        "--full-analysis", action="store_true",
        help="Run comprehensive analysis across all available DCAF signal runs",
    )
    parser.add_argument(
        "--delta-scale", type=float, default=DELTA_SCALE_DEFAULT,
        help=f"Scale factor for safety delta (default: {DELTA_SCALE_DEFAULT})",
    )

    # Classifier options
    parser.add_argument(
        "--classifier-model", type=str, default=None,
        help="Override classifier model (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--classifier-full-precision", action="store_true",
        help="Load classifier in bfloat16 instead of 8-bit quantization",
    )
    parser.add_argument(
        "--classifier-4bit", action="store_true",
        help="Load classifier in 4-bit quantization for maximum VRAM savings",
    )

    # Circuit analysis
    parser.add_argument(
        "--circuit-analysis", action="store_true",
        help="Enable circuit identification (works on any training run)",
    )

    # Probe configuration (for circuit analysis)
    parser.add_argument(
        "--probe-size", type=int, default=100,
        help="Number of probe prompts for analysis (default: 100)",
    )
    # Backward-compatible alias (hidden)
    parser.add_argument(
        "--circuit-probe-size", type=int, dest="probe_size",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-category", type=str, default=None,
        help="Harm category for probes (default: training run's category, or all)",
    )
    # Backward-compatible alias (hidden)
    parser.add_argument(
        "--circuit-category", type=str, dest="probe_category",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["recognition", "generation", "both"],
        default="both",
        help="Type of probes: 'recognition', 'generation', or 'both' (default: both)",
    )
    # Backward-compatible alias (hidden)
    parser.add_argument(
        "--circuit-probe-type", type=str, dest="probe_type",
        choices=["recognition", "generation", "both"],
        help=argparse.SUPPRESS,
    )

    # Circuit clustering configuration
    parser.add_argument(
        "--circuit-cluster", type=str, default="disjoint",
        choices=["disjoint", "probe-response", "functional"],
        help="Circuit clustering method (default: disjoint)",
    )
    parser.add_argument(
        "--circuit-edge-threshold", type=float, default=TAU_EDGE,
        help=f"Activation change threshold for circuit edge inclusion (default: {TAU_EDGE})",
    )
    parser.add_argument(
        "--circuit-attention-weight", type=float, default=ATTENTION_WEIGHT,
        help=f"Weight factor for attention-based edges (default: {ATTENTION_WEIGHT})",
    )

    # Domain analysis (spec-defined confidence computations)
    domain_group = parser.add_argument_group("Domain Analysis")
    domain_group.add_argument(
        "--weight", action="store_true",
        help="Run weight domain analysis (C_W computation)",
    )
    domain_group.add_argument(
        "--activation", action="store_true",
        help="Run activation domain analysis (C_A computation)",
    )
    domain_group.add_argument(
        "--geometry", action="store_true",
        help="Run geometry domain analysis (C_G computation)",
    )

    # Domain thresholds
    domain_group.add_argument(
        "--weight-confidence", type=float, default=TAU_W_DEFAULT,
        dest="tau_W",
        help=f"Weight confidence threshold (default: {TAU_W_DEFAULT})",
    )
    domain_group.add_argument(
        "--activation-confidence", type=float, default=TAU_A_DEFAULT,
        dest="tau_A",
        help=f"Activation confidence threshold (default: {TAU_A_DEFAULT})",
    )
    domain_group.add_argument(
        "--geometry-confidence", type=float, default=TAU_G_DEFAULT,
        dest="tau_G",
        help=f"Geometry confidence threshold (default: {TAU_G_DEFAULT})",
    )

    # Domain options
    domain_group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for model computation (default: auto-detect)",
    )
    domain_group.add_argument(
        "--skip-activation", action="store_true",
        help="Skip activation domain in --full (faster, weight+geometry only)",
    )
    domain_group.add_argument(
        "--skip-geometry", action="store_true",
        help="Skip geometry domain in --full (faster, weight+activation only)",
    )
    domain_group.add_argument(
        "--skip-circuit", action="store_true",
        help="Skip circuit analysis in --full (impact, classification, edges)",
    )
    domain_group.add_argument(
        "--top-k", type=int, default=TOP_K_CANDIDATES,
        help=f"Number of top candidates to return (default: {TOP_K_CANDIDATES})",
    )

    return parser


def run_analyze(args):
    """Run analysis with pre-parsed arguments."""
    # Import modules
    from dcaf.storage.delta_store import DeltaStore
    from dcaf.cli._analyze.utils import list_available_deltas

    # Resolve device
    device = detect_device(getattr(args, 'device', 'auto'))

    # Cross-run analysis mode
    if args.cross_run:
        # Default delta to compare across runs
        delta_name = getattr(args, 'delta_name', None) or "delta_t1_prefopt_target"
        logger.info(f"Cross-run analysis using delta: {delta_name}")

        # Inline cross-run: load weight analysis results per run and find stable core
        from dcaf.cli._analyze.weight_runner import run_weight_analysis
        run_results = {}
        for run_dir in args.cross_run:
            rp = Path(run_dir)
            if not rp.exists():
                logger.warning(f"Run directory not found: {rp}")
                continue
            run_results[str(rp)] = run_weight_analysis(
                run_path=rp,
                tau_W=args.tau_W if hasattr(args, "tau_W") else 0.5,
                top_k=200,
            )

        # Find parameters present in all runs (stable core)
        if run_results:
            param_sets = [
                {c["param_name"] for c in r.get("top_candidates", []) if c.get("param_name")}
                for r in run_results.values()
            ]
            stable_core = set.intersection(*param_sets) if param_sets else set()

            result = {
                "runs": list(run_results.keys()),
                "stable_core": sorted(stable_core),
                "stable_core_count": len(stable_core),
                "min_consistency": args.min_consistency,
            }

            logger.info(f"\nStable core: {len(stable_core)} parameters across {len(run_results)} runs")

            if args.output and result:
                output_path = Path(args.output)
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"\nResults saved to: {output_path}")

        return

    # Validate --run is provided for normal mode
    if not args.run:
        logger.error("--run is required (unless using --cross-run)")
        sys.exit(1)

    # Load run
    run_path = Path(args.run)
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_path}")
        sys.exit(1)

    delta_store = DeltaStore(run_path)

    if not delta_store.exists():
        logger.error(f"No metadata found in: {run_path}")
        sys.exit(1)

    # Load metadata
    metadata = delta_store.load_metadata()

    logger.info("=" * 60)
    logger.info("DCAF Analysis")
    logger.info("=" * 60)
    logger.info(f"Run: {run_path}")
    logger.info(f"Model: {metadata.model_name}")
    logger.info(f"Variant: {metadata.variant_name}")
    logger.info(f"Available deltas: {len(metadata.available_deltas)}")
    for d in metadata.available_deltas:
        logger.info(f"  - {d}")
    logger.info("")

    # List deltas mode
    if getattr(args, 'list_deltas', False):
        list_available_deltas(metadata.available_deltas)
        return

    # Refusal testing mode
    if args.test_refusal:
        results = test_refusal_rates(
            run_path,
            args.refusal_prompts,
            classifier_model=args.classifier_model,
            classifier_8bit=not args.classifier_full_precision,
            classifier_4bit=args.classifier_4bit,
        )
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to: {output_path}")
        return

    # Domain analysis modes (spec-defined confidence computations)
    if args.weight:
        from dcaf.cli._analyze.weight_runner import run_weight_analysis, display_weight_results
        results = run_weight_analysis(
            run_path=run_path,
            tau_W=args.tau_W,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        display_weight_results(results)
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to: {output_path}")
        return

    if args.activation:
        from dcaf.cli._analyze.activation_runner import run_activation_analysis, display_activation_results
        results = run_activation_analysis(
            run_path=run_path,
            model_name=metadata.model_name,
            tau_A=args.tau_A,
            probe_size=args.probe_size,
            probe_type=args.probe_type,
            top_k=args.top_k,
            device=device,
        )
        display_activation_results(results)
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to: {output_path}")
        return

    if args.geometry:
        from dcaf.cli._analyze.geometry_runner import run_geometry_analysis, display_geometry_results
        results = run_geometry_analysis(
            run_path=run_path,
            model_name=metadata.model_name,
            tau_G=args.tau_G,
            probe_size=args.probe_size,
            top_k=args.top_k,
            device=device,
        )
        display_geometry_results(results)
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to: {output_path}")
        return

    # Default: Full domain confidence analysis (C_W + C_A + C_G + triangulation)
    from dcaf.cli._analyze.full_runner import run_full_analysis, display_full_results
    results = run_full_analysis(
        run_path=run_path,
        model_name=metadata.model_name,
        tau_W=args.tau_W,
        tau_A=args.tau_A,
        tau_G=args.tau_G,
        probe_size=args.probe_size,
        top_k=args.top_k,
        skip_activation=getattr(args, 'skip_activation', False),
        skip_geometry=getattr(args, 'skip_geometry', False),
        skip_circuit=getattr(args, 'skip_circuit', False),
        device=device,
    )
    display_full_results(results)

    # SAE validation
    if args.validate_sae:
        logger.info(f"\nValidating with SAE features...")
        try:
            from dcaf.cli._analyze.sae_validator import SAEValidator

            validator = SAEValidator(metadata.model_name, load_sae=True)
            if validator.sae_loaded:
                matching_params = [p["param_name"] for p in results.get("top_candidates", []) if p.get("param_name")]
                sae_result = validator.validate_circuit(set(matching_params))
                logger.info(f"\nSAE Validation Results:")
                logger.info(f"  Correlation Score: {sae_result.correlation_score:.2f}")
                logger.info(f"  Validated: {sae_result.is_validated}")
                logger.info(f"  Safety Features Found: {len(sae_result.correlated_features)}")
                logger.info(f"  Explanation: {sae_result.explanation}")
            else:
                logger.warning("SAE not loaded - validation skipped")
        except ImportError as e:
            logger.warning(f"SAE validation unavailable: {e}")
        except Exception as e:
            logger.warning(f"SAE validation failed: {e}")

    # Ablation testing
    ablation_results = None
    matching_params = [p["param_name"] for p in results.get("top_candidates", []) if p.get("param_name")]

    if args.ablation and matching_params:
        # Parse comma-separated delta names
        safety_deltas = [d.strip() for d in args.ablation_deltas.split(",")]

        ablation_results = run_ablation_testing(
            run_path=run_path,
            matching_params=matching_params,
            top_k=args.ablation_top_k,
            prompts_per_param=args.ablation_prompts,
            safety_deltas=safety_deltas,
            classifier_model=args.classifier_model,
            classifier_8bit=not args.classifier_full_precision,
            classifier_4bit=args.classifier_4bit,
            enable_multi_probe=args.multi_probe_ablation,
            classification_threshold=args.classification_threshold,
            separation_ratio=args.separation_ratio,
            safe_prefix=args.safe_prefix,
            unsafe_prefix=args.unsafe_prefix,
        )
        display_ablation_results(
            ablation_results,
            filter_classification=args.filter_classification,
        )

    # Advanced ablation strategies (via runner functions)
    pair_results = None
    binary_results = None
    baseline_results = None

    if args.validate_baselines:
        baseline_results = run_baseline_validation(
            run_path=run_path,
            delta_scale=args.delta_scale,
            num_prompts=args.ablation_prompts,
            test_known_pairs=args.test_known_pairs,
        )
        if baseline_results and not baseline_results.get("all_passed"):
            logger.error("Baseline validation FAILED - check setup before running ablation")

    if args.ablation_pairs and matching_params:
        pair_results = run_pair_ablation(
            run_path=run_path,
            matching_params=matching_params,
            top_k=args.ablation_top_k,
            delta_scale=args.delta_scale,
            num_prompts=args.ablation_prompts,
        )

    if args.ablation_binary and matching_params:
        binary_results = run_binary_ablation(
            run_path=run_path,
            matching_params=matching_params,
            top_k=args.ablation_top_k,
            delta_scale=args.delta_scale,
            num_prompts=args.ablation_prompts,
        )

    # Group ablation (new)
    if args.group_ablation and matching_params:
        group_results = run_group_ablation(
            run_path=run_path,
            matching_params=matching_params,
            top_k=args.ablation_top_k,
            delta_scale=args.delta_scale,
            num_prompts=args.ablation_prompts,
        )

    # Circuit analysis
    circuit_results = None
    if args.circuit_analysis and matching_params:
        # Pass weight_classifications from ablation if available
        weight_classifications = None
        if ablation_results and ablation_results.weight_classifications:
            weight_classifications = ablation_results.weight_classifications

        circuit_results = run_probe_analysis(
            run_path=run_path,
            matching_params=matching_params,
            model_name=metadata.model_name,
            cluster_method=args.circuit_cluster,
            edge_threshold=args.circuit_edge_threshold,
            attention_weight=args.circuit_attention_weight,
            probe_type=args.probe_type,
            probe_size=args.probe_size,
            category=args.probe_category,
            weight_classifications=weight_classifications,
        )

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_data = dict(results)

        # Include ablation results if available
        if ablation_results is not None:
            output_data["ablation_results"] = ablation_results.to_dict()

        # Include advanced ablation results
        if pair_results is not None:
            output_data["pair_ablation_results"] = {
                "total_pairs": len(pair_results.all_results),
                "breaking_pairs": len(pair_results.breaking_pairs),
                "break_rate": pair_results.break_rate,
                "pairs": [
                    {
                        "param1": r.param1,
                        "param2": r.param2,
                        "harmful_count": r.harmful_count,
                        "total_count": r.total_count,
                        "is_breaking": r.is_breaking,
                    }
                    for r in pair_results.all_results
                ],
            }

        if binary_results is not None:
            output_data["binary_search_results"] = {
                "initial_count": len(binary_results.initial_params),
                "critical_count": len(binary_results.critical_params),
                "reduction_ratio": binary_results.reduction_ratio,
                "critical_params": binary_results.critical_params,
                "iterations": binary_results.iterations,
                "search_log": binary_results.search_log,
            }

        if baseline_results is not None:
            output_data["baseline_validation"] = {
                "all_passed": baseline_results.all_passed,
                "base_model": {
                    "passed": baseline_results.base_model.passed,
                    "harm_rate": baseline_results.base_model.harm_rate,
                },
                "safe_model": {
                    "passed": baseline_results.safe_model.passed,
                    "harm_rate": baseline_results.safe_model.harm_rate,
                },
                "known_pairs": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "expected_harmful": r.expected_harmful,
                        "actual_harmful": r.actual_harmful,
                    }
                    for r in baseline_results.known_pairs
                ],
            }

        # Circuit analysis results
        if circuit_results is not None:
            output_data["circuit_analysis"] = circuit_results

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path}")

    # Generate visualization if requested
    if args.visualize or args.show_chart:
        from dcaf.cli._analyze.visualize import create_distribution_charts
        create_distribution_charts(
            results,
            output_path=args.visualize,
            show=args.show_chart,
        )


def main():
    """Standalone entry point for dcaf-analyze."""
    parser = create_argparser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    run_analyze(args)


__all__ = ["create_argparser", "run_analyze", "main"]
