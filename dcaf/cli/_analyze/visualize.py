"""Visualization helpers for DCAF analysis results."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def _candidate_layer(candidate: Dict[str, Any]) -> Optional[int]:
    layer = candidate.get("layer")
    if isinstance(layer, int):
        return layer

    text = " ".join(
        str(candidate.get(key, ""))
        for key in ("param_name", "component", "id")
    )
    match = re.search(r"(?:layers?|h|blocks?)\.(\d+)\.", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bL(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _candidate_component(candidate: Dict[str, Any]) -> str:
    component = str(candidate.get("component") or candidate.get("param_name") or "")
    lower = component.lower()
    if "mlp" in lower or "gate_proj" in lower or "up_proj" in lower or "down_proj" in lower:
        return "MLP"
    if "attn" in lower or "self_attn" in lower or re.search(r"\bL\d+H", component):
        return "Attention"
    return "Other"


def _iter_candidates(results: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    candidates = results.get("top_candidates")
    if candidates is None and isinstance(results.get("full"), dict):
        candidates = results["full"].get("top_candidates")
    return candidates or []


def create_distribution_charts(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Create layer and component distribution charts for top candidates."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install with `pip install dcaf[visualization]`."
        ) from exc

    candidates = list(_iter_candidates(results))
    if not candidates:
        logger.warning("No top candidates available for visualization")
        return

    layer_counts: Dict[int, int] = {}
    component_counts: Dict[str, int] = {}
    for candidate in candidates:
        layer = _candidate_layer(candidate)
        if layer is not None:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        component = _candidate_component(candidate)
        component_counts[component] = component_counts.get(component, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("DCAF Candidate Distribution")

    if layer_counts:
        layers = sorted(layer_counts)
        axes[0].bar([str(layer) for layer in layers], [layer_counts[layer] for layer in layers])
        axes[0].set_title("By Layer")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Candidates")
    else:
        axes[0].text(0.5, 0.5, "No layer metadata", ha="center", va="center")
        axes[0].set_axis_off()

    labels = sorted(component_counts)
    axes[1].bar(labels, [component_counts[label] for label in labels])
    axes[1].set_title("By Component Type")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Candidates")

    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved distribution chart to {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = ["create_distribution_charts"]
