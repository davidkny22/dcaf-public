"""
Formatting functions for analysis output display.

Used by the CLI to present analysis results in a human-readable format.
"""

import logging
from typing import Any, Dict, Optional

from dcaf.ablation.results import AblationResults

logger = logging.getLogger(__name__)


def format_summary(result) -> str:
    """Format analysis result summary for display."""
    lines = []
    lines.append(f"Matching parameters: {result.matching_count}")
    lines.append("")

    # By layer
    if result.summary.get("by_layer"):
        lines.append("By layer:")
        for layer, count in sorted(result.summary["by_layer"].items(), key=lambda x: int(x[0])):
            lines.append(f"  Layer {layer}: {count}")
        lines.append("")

    # By component
    if result.summary.get("by_component"):
        lines.append("By component:")
        for comp, count in result.summary["by_component"].items():
            if count > 0:
                lines.append(f"  {comp}: {count}")
        lines.append("")

    return "\n".join(lines)


def format_params(param_details: list, max_show: int = 20) -> str:
    """Format parameter details for display."""
    lines = []

    shown = param_details[:max_show]
    for detail in shown:
        name = detail["name"]
        layer = detail.get("layer", "?")
        comp = detail.get("component", "?")
        lines.append(f"  [{layer}] {comp}: {name}")

    if len(param_details) > max_show:
        lines.append(f"  ... and {len(param_details) - max_show} more")

    return "\n".join(lines)


def compare_results(result_a, result_b) -> str:
    """Compare two analysis results."""
    lines = []

    set_a = set(result_a.matching_params)
    set_b = set(result_b.matching_params)

    only_a = set_a - set_b
    only_b = set_b - set_a
    both = set_a & set_b

    lines.append(f"\nComparison: {result_a.criteria_name} vs {result_b.criteria_name}")
    lines.append("=" * 60)
    lines.append(f"Only in {result_a.criteria_name}: {len(only_a)}")
    lines.append(f"Only in {result_b.criteria_name}: {len(only_b)}")
    lines.append(f"In both: {len(both)}")
    lines.append("")

    if only_a:
        lines.append(f"\nOnly in {result_a.criteria_name}:")
        for p in sorted(list(only_a)[:10]):
            lines.append(f"  {p}")
        if len(only_a) > 10:
            lines.append(f"  ... and {len(only_a) - 10} more")

    if only_b:
        lines.append(f"\nOnly in {result_b.criteria_name}:")
        for p in sorted(list(only_b)[:10]):
            lines.append(f"  {p}")
        if len(only_b) > 10:
            lines.append(f"  ... and {len(only_b) - 10} more")

    return "\n".join(lines)


def display_ablation_results(
    results: AblationResults,
    filter_classification: Optional[str] = None,
) -> None:
    """Display ablation results summary with multi-probe classification breakdown."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total tested:      {results.total_tested}")
    logger.info(f"Safety-specific:   {results.validated_count} ({results.validated_count/max(results.total_tested,1)*100:.1f}%)")
    logger.info(f"Not specific:      {results.rejected_count} ({results.rejected_count/max(results.total_tested,1)*100:.1f}%)")
    logger.info(f"Broke generation:  {results.skipped_count} ({results.skipped_count/max(results.total_tested,1)*100:.1f}%)")

    # Multi-probe classification breakdown
    if results.weight_classifications:
        logger.info("")
        logger.info("Classification Breakdown:")

        classification_counts = {
            "recognition-specific": 0,
            "generation-specific": 0,
            "shared": 0,
            "false-positive": 0,
        }

        for wc in results.weight_classifications.values():
            classification = wc.classification if hasattr(wc, 'classification') else wc.get('classification', 'unknown')
            if classification in classification_counts:
                classification_counts[classification] += 1

        for cat, count in classification_counts.items():
            pct = count / max(results.total_tested, 1) * 100
            logger.info(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")

        # Show filtered results if requested
        if filter_classification:
            filtered = []
            for r in results.param_results:
                if r.weight_classification and r.weight_classification.classification == filter_classification:
                    filtered.append(r)

            logger.info(f"\n{filter_classification.upper()} Weights ({len(filtered)}):")
            for r in filtered[:10]:
                wc = r.weight_classification
                logger.info(
                    f"  {r.param_name:60s} "
                    f"F:{wc.free_generation_impact:.2f} T:{wc.teacher_forcing_impact:.2f} "
                    f"ratio:{wc.separation_ratio:.1f}"
                )
            if len(filtered) > 10:
                logger.info(f"  ... and {len(filtered) - 10} more")

    # List validated params
    validated = [r for r in results.param_results if r.ablation_validated]
    if validated:
        logger.info("")
        logger.info(f"Validated safety-specific params ({len(validated)}):")
        for r in validated[:20]:
            if hasattr(r, 'weight_classification') and r.weight_classification:
                wc = r.weight_classification
                logger.info(f"  {r.param_name} ({wc.classification})")
            else:
                logger.info(f"  {r.param_name}")
        if len(validated) > 20:
            logger.info(f"  ... and {len(validated) - 20} more")


def format_signal_breakdown(
    signal_details: Dict[str, Dict[str, Any]],
    verbose: bool = False,
) -> str:
    """
    Format signal breakdown sorted by confidence.

    Args:
        signal_details: Dict mapping param/component names to signal info
        verbose: Whether to show detailed signal lists

    Returns:
        Formatted string showing signal-based confidence rankings

    Example output:
        Signal-Based Confidence Rankings:
          param1   11/11 signals (1.00) [OPP]
          param2   10/11 signals (0.91) [OPP]
          param3    8/11 signals (0.73)
          param4    5/11 signals (0.45)
    """
    if not signal_details:
        return "No signal details available"

    # Sort by confidence descending
    sorted_items = sorted(
        signal_details.items(),
        key=lambda x: x[1].get("relevance_confidence", 0.0),
        reverse=True,
    )

    lines = []
    lines.append("\nSignal-Based Confidence Rankings:")
    lines.append("=" * 80)

    for name, details in sorted_items:
        signal_count = details.get("signal_count", 0)
        confidence = details.get("relevance_confidence", 0.0)
        has_opposition = details.get("has_opposition", False)
        signals = details.get("signals", {})

        # Total signals = number of keys in signals dict (or infer from confidence)
        if signals:
            total_signals = len([k for k, v in signals.items() if v is not None])
            # Better: infer from signal_count and confidence
            if confidence > 0:
                total_signals = int(signal_count / confidence)
            else:
                total_signals = signal_count

        opp_marker = " [OPP]" if has_opposition else ""

        # Format: name   count/total (confidence) [OPP]
        line = f"  {name:50s} {signal_count:2d}/{total_signals:2d} signals ({confidence:.2f}){opp_marker}"
        lines.append(line)

        # Verbose: show which signals passed
        if verbose and signals:
            passing = [k for k, v in signals.items() if v]
            if passing:
                lines.append(f"    Signals: {', '.join(sorted(passing))}")

    return "\n".join(lines)


__all__ = [
    "format_summary",
    "format_params",
    "compare_results",
    "display_ablation_results",
    "format_signal_breakdown",
]
