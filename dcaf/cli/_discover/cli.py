"""
DCAF Discover CLI (§3 Multi-Path Discovery).

Coordinates the three discovery paths:
  H_W: Weight-based discovery — parameters that changed significantly
  H_A: Activation-based discovery — leverage points
  H_G: Gradient-based discovery — high behavioral gradients

Output: discovery.json saved to run directory.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Any, Optional, List

from dcaf.cli.common import (
    detect_device,
    configure_logging,
    add_run_path_args,
    add_discovery_threshold_args,
    add_model_args,
    add_device_args,
    add_probe_args,
    add_output_path_args,
    add_verbose_args,
)

logger = logging.getLogger(__name__)


def create_argparser() -> argparse.ArgumentParser:
    """Create argparser for discovery CLI."""
    parser = argparse.ArgumentParser(
        description="DCAF Discovery Phase - Identify candidate parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Discovery Paths:
  H_W  Weight-based: parameters that changed significantly during training
  H_A  Activation-based: leverage points (small delta-w, large delta-activation)
  H_G  Gradient-based: parameters with high behavioral gradients

Output:
  Creates discovery.json in the run directory with:
  - H_disc = H_W | H_A | H_G (union of all paths)
  - DiscoveryInfo for each parameter (paths, bonus, scores)
  - Summary statistics

Examples:
  # Weight-only discovery (fast, no model loading)
  dcaf discover -r ./runs/run_001/

  # Weight + activation discovery
  dcaf discover -r ./runs/run_001/ -a

  # All three paths
  dcaf discover -r ./runs/run_001/ -a -g

  # Custom thresholds
  dcaf discover -r ./runs/run_001/ --significance-threshold 90 --activation-threshold 80
""",
    )

    # Shared arguments
    add_run_path_args(parser)

    # Discovery paths (flags) -- unique to discover
    parser.add_argument(
        "--weight", "-w",
        dest="weight",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable H_W weight-based discovery (default: on)",
    )
    parser.add_argument(
        "--activation", "-a",
        action="store_true",
        default=False,
        help="Enable H_A activation-based discovery (requires model)",
    )
    parser.add_argument(
        "--gradient", "-g",
        action="store_true",
        default=False,
        help="Enable H_G gradient-based discovery (requires model + data)",
    )

    # Shared threshold, model, device, probe, output, verbose arguments
    add_discovery_threshold_args(parser)
    add_model_args(parser)
    add_device_args(parser)
    add_probe_args(parser, default=50)
    add_output_path_args(parser)
    add_verbose_args(parser)

    return parser


def run_discovery(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run multi-path discovery.

    Returns:
        Discovery results dict ready for JSON serialization
    """
    from dcaf.storage import DeltaStore

    # Resolve device
    device = detect_device(args.device)

    run_path = Path(args.run)
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_path}")
        return {"error": f"Run directory not found: {run_path}"}

    logger.info("=" * 60)
    logger.info("DCAF DISCOVERY PHASE")
    logger.info("=" * 60)
    logger.info(f"Run: {run_path}")
    logger.info(f"Paths: W={args.weight}, A={args.activation}, G={args.gradient}")

    # Load metadata
    delta_store = DeltaStore(run_path)
    metadata = delta_store.load_metadata()

    model_name = args.model
    if model_name is None and metadata:
        model_name = getattr(metadata, 'model_name', None)

    # Initialize results
    H_W: Set[Any] = set()
    H_A: Set[Any] = set()
    H_G: Set[Any] = set()
    S_W: Dict[Any, float] = {}
    S_A: Dict[Any, float] = {}
    S_G: Dict[Any, float] = {}
    param_names_list: Optional[List[str]] = None

    # ========== H_W: Weight Discovery ==========
    if args.weight:
        logger.info("\n" + "-" * 60)
        logger.info("PATH 1: Weight-Based Discovery (H_W)")
        logger.info("-" * 60)

        from dcaf.cli._discover.weight_discovery import run_weight_discovery

        H_W, S_W, w_names = run_weight_discovery(
            run_path=run_path,
            tau_sig=args.tau_sig,
            tau_base=args.tau_base,
        )
        if param_names_list is None:
            param_names_list = w_names
        logger.info(f"H_W: {len(H_W)} parameters discovered")

    # ========== H_A: Activation Discovery ==========
    if args.activation:
        logger.info("\n" + "-" * 60)
        logger.info("PATH 2: Activation-Based Discovery (H_A)")
        logger.info("-" * 60)

        if model_name is None:
            logger.error("Model name required for H_A. Use --model or ensure metadata.")
            return {"error": "Model name required for H_A"}

        from dcaf.cli._discover.activation_discovery import run_activation_discovery

        H_A, S_A, a_names = run_activation_discovery(
            run_path=run_path,
            model_name=model_name,
            tau_comp=args.tau_comp,
            tau_act=args.tau_act,
            probe_size=args.probe_size,
            device=device,
        )
        if param_names_list is None:
            param_names_list = a_names
        logger.info(f"H_A: {len(H_A)} parameters discovered")

    # ========== H_G: Gradient Discovery ==========
    if args.gradient:
        logger.info("\n" + "-" * 60)
        logger.info("PATH 3: Gradient-Based Discovery (H_G)")
        logger.info("-" * 60)

        if model_name is None:
            logger.error("Model name required for H_G. Use --model or ensure metadata.")
            return {"error": "Model name required for H_G"}

        from dcaf.cli._discover.gradient_discovery import run_gradient_discovery

        H_G, S_G, g_names = run_gradient_discovery(
            run_path=run_path,
            model_name=model_name,
            tau_grad=args.tau_grad,
            device=device,
        )
        if param_names_list is None:
            param_names_list = g_names
        logger.info(f"H_G: {len(H_G)} parameters discovered")

    # ========== Integration ==========
    logger.info("\n" + "-" * 60)
    logger.info("INTEGRATION: H_disc = H_W | H_A | H_G")
    logger.info("-" * 60)

    from dcaf.cli._discover.integration import (
        create_discovery_output,
        save_discovery_result,
    )

    result = create_discovery_output(
        H_W=H_W,
        H_A=H_A,
        H_G=H_G,
        S_W=S_W,
        S_A=S_A,
        S_G=S_G,
        run_path=run_path,
        config={
            "tau_sig": args.tau_sig,
            "tau_base": args.tau_base,
            "tau_act": args.tau_act,
            "tau_grad": args.tau_grad,
            "tau_comp": args.tau_comp,
            "beta_path": args.beta_path,
        },
        beta_path=args.beta_path,
        param_names=param_names_list,
    )

    # Save
    output_path = Path(args.output) if args.output else run_path / "discovery.json"
    save_discovery_result(result, output_path)

    # Summary
    summary = result.get("summary", {})
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total discovered: {summary.get('total_discovered', 0)}")
    logger.info(f"By path: H_W={summary.get('by_path', {}).get('H_W', 0)}, "
                f"H_A={summary.get('by_path', {}).get('H_A', 0)}, "
                f"H_G={summary.get('by_path', {}).get('H_G', 0)}")
    logger.info(f"Multi-path: {summary.get('multi_path_count', 0)}")
    logger.info(f"Saved to: {output_path}")

    return result


def main():
    """Main entry point for discover CLI."""
    parser = create_argparser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    result = run_discovery(args)

    if "error" in result:
        sys.exit(1)


__all__ = ["create_argparser", "run_discovery", "main"]


if __name__ == "__main__":
    main()
