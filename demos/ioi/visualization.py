"""IOI circuit discovery visualization.

Two-panel matplotlib figure:
  Panel 1: Circuit diagram — grid of components with discovered/known overlay
  Panel 2: Confidence breakdown — stacked bar chart of top components
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

ROLE_COLORS = {
    "recognition": "#4A90D9",
    "steering": "#50C878",
    "preference": "#E74C3C",
}
DEFAULT_COLOR = "#999999"


def plot_circuit_diagram(
    components: List[Dict[str, Any]],
    known_heads: Dict[str, Dict],
    edges: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    n_layers: int = 12,
    n_heads: int = 12,
):
    """Create two-panel circuit discovery visualization.

    Args:
        components: List of dicts with keys: id, C_unified, C_W, C_A, C_G,
                   paths, function (optional), bonus (optional)
        known_heads: Dict mapping component ID to role info (from known_circuit.py)
        edges: Optional list of dicts with keys: source, target, weight
        output_path: Save figure to this path (creates parent dirs)
        show: Call plt.show() for interactive display
        n_layers: Number of layers in the model
        n_heads: Number of attention heads per layer
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [3, 2]})

    _draw_circuit_grid(ax1, components, known_heads, edges, n_layers, n_heads)
    _draw_confidence_bars(ax2, components)

    fig.suptitle("DCAF IOI Circuit Discovery — GPT-2 Small", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved circuit diagram to {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_component_id(comp_id: str):
    """Parse 'L9H9' -> (9, 9, 'attn') or 'L10_MLP' -> (10, None, 'mlp')."""
    if "_MLP" in comp_id:
        layer = int(comp_id.split("_")[0][1:])
        return layer, None, "mlp"
    elif "H" in comp_id:
        parts = comp_id.replace("L", "").split("H")
        return int(parts[0]), int(parts[1]), "attn"
    return None, None, None


def _draw_circuit_grid(ax, components, known_heads, edges, n_layers, n_heads):
    """Panel 1: Grid layout with discovered and known components."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-1.5, n_heads - 0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Head", fontsize=12)
    ax.set_title("Circuit Diagram", fontsize=14)
    ax.set_xticks(range(n_layers))
    ax.set_yticks(list(range(n_heads)) + [-1])
    ax.set_yticklabels([str(i) for i in range(n_heads)] + ["MLP"])
    ax.grid(True, alpha=0.15)
    ax.invert_yaxis()

    for comp_id, info in known_heads.items():
        layer, head, comp_type = _parse_component_id(comp_id)
        if layer is None:
            continue
        y = head if comp_type == "attn" else -1
        ax.plot(
            layer, y, "s",
            markersize=18, markeredgecolor="black",
            markeredgewidth=1.5, markerfacecolor="none",
            linestyle="--", alpha=0.5,
        )

    for comp in components:
        layer, head, comp_type = _parse_component_id(comp["id"])
        if layer is None:
            continue
        y = head if comp_type == "attn" else -1
        c_unified = comp.get("C_unified", 0.3)
        func = comp.get("function", None)
        color = ROLE_COLORS.get(func, DEFAULT_COLOR)
        size = 80 + c_unified * 400

        ax.scatter(
            layer, y, s=size, c=color, alpha=0.8,
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        ax.annotate(
            comp["id"], (layer, y),
            textcoords="offset points", xytext=(0, -15),
            fontsize=6, ha="center", alpha=0.7,
        )

    if edges:
        for edge in edges:
            src_layer, src_head, src_type = _parse_component_id(edge["source"])
            tgt_layer, tgt_head, tgt_type = _parse_component_id(edge["target"])
            if src_layer is None or tgt_layer is None:
                continue
            src_y = src_head if src_type == "attn" else -1
            tgt_y = tgt_head if tgt_type == "attn" else -1
            weight = edge.get("weight", 0.5)
            ax.annotate(
                "", xy=(tgt_layer, tgt_y), xytext=(src_layer, src_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    lw=0.5 + weight * 2,
                    alpha=0.4 + weight * 0.4,
                ),
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["recognition"],
               markersize=10, label="Recognition"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["steering"],
               markersize=10, label="Steering"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["preference"],
               markersize=10, label="Preference"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DEFAULT_COLOR,
               markersize=10, label="Unclassified"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor="black",
               markerfacecolor="none", markersize=10, label="Known IOI head"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)


def _draw_confidence_bars(ax, components, top_n: int = 15):
    """Panel 2: Stacked horizontal bar chart of confidence breakdown."""
    import numpy as np

    sorted_comps = sorted(components, key=lambda c: c.get("C_unified", 0), reverse=True)[:top_n]
    sorted_comps = sorted_comps[::-1]

    labels = [c["id"] for c in sorted_comps]
    c_w = [c.get("C_W", 0) for c in sorted_comps]
    c_a = [c.get("C_A", 0) for c in sorted_comps]
    c_g = [c.get("C_G", 0) for c in sorted_comps]
    bonus = [c.get("bonus", 0) for c in sorted_comps]

    y = np.arange(len(labels))
    h = 0.6

    ax.barh(y, c_w, h, label="C_W (Weight)", color="#4A90D9", alpha=0.85)
    ax.barh(y, c_a, h, left=c_w, label="C_A (Activation)", color="#F5A623", alpha=0.85)
    left2 = [w + a for w, a in zip(c_w, c_a)]
    ax.barh(y, c_g, h, left=left2, label="C_G (Geometry)", color="#50C878", alpha=0.85)

    if any(b > 0 for b in bonus):
        left3 = [w + a + g for w, a, g in zip(c_w, c_a, c_g)]
        ax.barh(y, bonus, h, left=left3, label="Multi-path bonus",
                color="#E74C3C", alpha=0.5, hatch="//")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_title("Top Component Confidence Breakdown", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, label="tau_unified")
