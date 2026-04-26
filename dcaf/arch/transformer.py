"""Transformer architecture utilities (§15).

Parameter parsing, component resolution, and exclusion patterns
for transformer-based models (LLaMA, Gemma, GPT-2, Pythia).

Implements:
  - Remark 1.3 (Excluded from Analysis): exclusion of embeddings, layer norms,
    and positional encodings from the projection set P.
  - §15.2 (Component Decomposition): mapping parameter names to layer/component
    metadata and resolving component IDs to parameter name lists.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


EXCLUDED_PARAM_PATTERNS = [
    "embed_out", "embed_in", "wte", "wpe", "lm_head",
    "ln_f", "ln_1", "ln_2", "layer_norm", "layernorm",
    "final_layer_norm", "input_layernorm", "post_attention_layernorm",
    "position", "pos_emb", "rotary",
]


def should_exclude_param(param_name: str) -> bool:
    """Check if a parameter should be excluded from circuit analysis (Remark 1.3).

    Excludes: embeddings, layer norms, positional encodings — shared
    input/output transformations, not behavioral circuit components.
    """
    param_lower = param_name.lower()
    return any(pattern in param_lower for pattern in EXCLUDED_PARAM_PATTERNS)


LAYER_PATTERNS = [
    r"layers?\.(\d+)\.",
    r"h\.(\d+)\.",
    r"transformer\.h\.(\d+)\.",
    r"blocks?\.(\d+)\.",
    r"gpt_neox\.layers\.(\d+)\.",
]

COMPONENT_PATTERNS = {
    "attention": [
        r"attention", r"attn", r"self_attn",
        r"query", r"key", r"value", r"q_proj", r"k_proj", r"v_proj",
        r"dense", r"out_proj", r"c_attn", r"c_proj",
    ],
    "mlp": [
        r"mlp", r"feed_forward", r"ffn",
        r"fc1", r"fc2", r"dense_h_to_4h", r"dense_4h_to_h",
        r"c_fc", r"gate_proj", r"up_proj", r"down_proj",
    ],
}


def parse_param_metadata(param_name: str) -> Dict[str, Optional[Any]]:
    """Extract layer number and component type from a parameter name."""
    layer = None
    component = None
    param_lower = param_name.lower()

    for pattern in LAYER_PATTERNS:
        match = re.search(pattern, param_lower)
        if match:
            layer = int(match.group(1))
            break

    for comp_type, patterns in COMPONENT_PATTERNS.items():
        if any(p in param_lower for p in patterns):
            component = comp_type
            break

    if component is None and layer is not None:
        component = "other"

    return {"layer": layer, "component": component}


def get_component_params(component: str, all_params: List[str]) -> List[str]:
    """Resolve a component ID to its parameter names.

    Component ID formats (Remark 15.2):
      L{n}_MLP  -> model.layers.{n}.mlp.*
      L{n}H{h}  -> model.layers.{n}.self_attn.*
    """
    if "_MLP" in component:
        layer_num = component.split("_")[0][1:]
        prefix = f"model.layers.{layer_num}.mlp."
        return [p for p in all_params if prefix in p]
    elif component.startswith("L") and "H" in component:
        parts = component[1:].split("H")
        layer_num = parts[0]
        prefix = f"model.layers.{layer_num}.self_attn."
        return [p for p in all_params if prefix in p]
    else:
        return [p for p in all_params if component in p]


def get_param_summary(param_names: List[str]) -> Tuple[Dict[int, int], Dict[str, int]]:
    """Summarize parameters by layer and component type."""
    by_layer: Dict[int, int] = {}
    by_component: Dict[str, int] = {}

    for name in param_names:
        meta = parse_param_metadata(name)
        if meta["layer"] is not None:
            by_layer[meta["layer"]] = by_layer.get(meta["layer"], 0) + 1
        if meta["component"] is not None:
            by_component[meta["component"]] = by_component.get(meta["component"], 0) + 1

    return by_layer, by_component


__all__ = [
    "EXCLUDED_PARAM_PATTERNS",
    "LAYER_PATTERNS",
    "COMPONENT_PATTERNS",
    "should_exclude_param",
    "parse_param_metadata",
    "get_component_params",
    "get_param_summary",
]
