"""
Model Topology — projection and component enumeration.

Maps model parameters to the two-level hierarchy:
  Projection: individual weight matrix (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down)
  Component: functional unit (attention head, MLP layer)

Supports:
  - LLaMA-family (Gemma, LLaMA, Mistral, Qwen) with separate Q/K/V/O projections
  - GQA (grouped-query attention) with shared K/V projections
  - GPT-2/GPT-Neo with fused QKV (c_attn)
  - Pythia/GPT-NeoX with fused query_key_value
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from torch import Tensor

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProjectionSlice:
    """How to extract a projection's weight matrix from a model parameter."""
    param_name: str        # e.g., "model.layers.5.self_attn.q_proj.weight"
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_end)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_start, self.col_end)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row_end - self.row_start, self.col_end - self.col_start)


@dataclass
class ModelTopology:
    """Complete projection-to-component mapping for a model."""

    # All unique projection IDs (analysis units, ~700 for Gemma 2 2B)
    projections: List[str]

    # All component IDs (graph nodes, ~234 for Gemma 2 2B)
    components: List[str]

    # Projection -> set of components it belongs to.
    # Most projections map to one component; shared K/V map to multiple.
    proj_to_components: Dict[str, Set[str]]

    # Component -> list of projection IDs it contains.
    component_to_projs: Dict[str, List[str]]

    # Extraction recipe per projection (param name + slice bounds).
    proj_slices: Dict[str, ProjectionSlice]

    # Model architecture parameters
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    architecture: str  # "llama", "gpt2", "pythia"

    @property
    def gqa_group_size(self) -> int:
        """Number of query heads per KV group."""
        return self.n_query_heads // self.n_kv_heads

    @property
    def is_gqa(self) -> bool:
        """True if model uses grouped-query attention."""
        return self.n_query_heads != self.n_kv_heads

    def get_projection_shape(self, proj_id: str) -> Tuple[int, int]:
        """Return (rows, cols) for a projection."""
        return self.proj_slices[proj_id].shape


# =============================================================================
# Architecture Detection
# =============================================================================

def _detect_architecture(model) -> str:
    """Detect model architecture family from structure."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return "pythia"
    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Expected LLaMA-family, GPT-2, or Pythia/GPT-NeoX."
    )


def _get_n_layers(config) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine number of layers from model config")


def _get_n_query_heads(config) -> int:
    for attr in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine number of query heads from model config")


def _get_n_kv_heads(config) -> int:
    for attr in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                return val
    # No GQA — KV heads = query heads
    return _get_n_query_heads(config)


def _get_head_dim(config) -> int:
    if hasattr(config, "head_dim") and config.head_dim is not None:
        return config.head_dim
    return config.hidden_size // _get_n_query_heads(config)


def _get_intermediate_size(config) -> int:
    for attr in ("intermediate_size", "n_inner", "ffn_dim"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                return val
    # GPT-2 default: 4 * hidden_size
    return 4 * config.hidden_size


# =============================================================================
# Param Name Resolution
# =============================================================================

def _llama_attn_param(layer: int, proj: str) -> str:
    """LLaMA-family attention parameter name."""
    return f"model.layers.{layer}.self_attn.{proj}_proj.weight"


def _llama_mlp_param(layer: int, proj: str) -> str:
    """LLaMA-family MLP parameter name."""
    return f"model.layers.{layer}.mlp.{proj}_proj.weight"


def _gpt2_attn_param(layer: int) -> str:
    """GPT-2 fused QKV parameter name."""
    return f"transformer.h.{layer}.attn.c_attn.weight"


def _gpt2_attn_out_param(layer: int) -> str:
    return f"transformer.h.{layer}.attn.c_proj.weight"


def _gpt2_mlp_fc_param(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_fc.weight"


def _gpt2_mlp_proj_param(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def _pythia_qkv_param(layer: int) -> str:
    """Pythia/GPT-NeoX fused QKV parameter name."""
    return f"gpt_neox.layers.{layer}.attention.query_key_value.weight"


def _pythia_attn_out_param(layer: int) -> str:
    return f"gpt_neox.layers.{layer}.attention.dense.weight"


def _pythia_mlp_up_param(layer: int) -> str:
    return f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight"


def _pythia_mlp_down_param(layer: int) -> str:
    return f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight"


# =============================================================================
# Projection Builders
# =============================================================================

def _add_projection(
    proj_id: str,
    component_ids: Set[str],
    pslice: ProjectionSlice,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Register one projection with its component mapping."""
    projections.append(proj_id)
    proj_to_components[proj_id] = component_ids
    proj_slices[proj_id] = pslice
    for comp in component_ids:
        component_to_projs[comp].append(proj_id)


def _build_llama_attention(
    layer: int,
    n_query_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build attention projections for LLaMA-family (separate Q/K/V/O params)."""
    group_size = n_query_heads // n_kv_heads

    for h in range(n_query_heads):
        comp = f"L{layer}H{h}"

        # Q projection — per query head, sliced from q_proj
        q_id = f"L{layer}H{h}_Q"
        _add_projection(
            q_id, {comp},
            ProjectionSlice(
                param_name=_llama_attn_param(layer, "q"),
                row_start=h * head_dim, row_end=(h + 1) * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # O projection — per query head, sliced from o_proj columns
        o_id = f"L{layer}H{h}_O"
        _add_projection(
            o_id, {comp},
            ProjectionSlice(
                param_name=_llama_attn_param(layer, "o"),
                row_start=0, row_end=hidden_size,
                col_start=h * head_dim, col_end=(h + 1) * head_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

    # K/V projections — per KV group, shared across query heads in the group
    for g in range(n_kv_heads):
        # Components sharing this KV group
        shared_comps = {f"L{layer}H{h}" for h in range(g * group_size, (g + 1) * group_size)}

        if group_size == 1:
            # MHA: name as per-head for simplicity
            k_id = f"L{layer}H{g}_K"
            v_id = f"L{layer}H{g}_V"
        else:
            # GQA: explicit KV group naming
            k_id = f"L{layer}KV{g}_K"
            v_id = f"L{layer}KV{g}_V"

        _add_projection(
            k_id, shared_comps,
            ProjectionSlice(
                param_name=_llama_attn_param(layer, "k"),
                row_start=g * head_dim, row_end=(g + 1) * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        _add_projection(
            v_id, shared_comps,
            ProjectionSlice(
                param_name=_llama_attn_param(layer, "v"),
                row_start=g * head_dim, row_end=(g + 1) * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )


def _build_llama_mlp(
    layer: int,
    hidden_size: int,
    intermediate_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build MLP projections for LLaMA-family (gate/up/down, SwiGLU/GeGLU)."""
    comp = f"L{layer}_MLP"

    for proj_name, param_suffix, out_dim, in_dim in [
        ("gate", "gate", intermediate_size, hidden_size),
        ("up", "up", intermediate_size, hidden_size),
        ("down", "down", hidden_size, intermediate_size),
    ]:
        proj_id = f"L{layer}_MLP_{proj_name}"
        _add_projection(
            proj_id, {comp},
            ProjectionSlice(
                param_name=_llama_mlp_param(layer, param_suffix),
                row_start=0, row_end=out_dim,
                col_start=0, col_end=in_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )


def _build_gpt2_attention(
    layer: int,
    n_heads: int,
    head_dim: int,
    hidden_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build attention projections for GPT-2 (fused c_attn = [Q|K|V])."""
    # GPT-2 c_attn.weight: [hidden_size, 3 * hidden_size]
    # Layout: columns [0:H] = Q, [H:2H] = K, [2H:3H] = V
    fused_param = _gpt2_attn_param(layer)
    out_param = _gpt2_attn_out_param(layer)

    for h in range(n_heads):
        comp = f"L{layer}H{h}"

        # Q — from fused c_attn columns [h*hd : (h+1)*hd]
        _add_projection(
            f"L{layer}H{h}_Q", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=0, row_end=hidden_size,
                col_start=h * head_dim, col_end=(h + 1) * head_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # K — from fused c_attn columns [H + h*hd : H + (h+1)*hd]
        k_offset = hidden_size
        _add_projection(
            f"L{layer}H{h}_K", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=0, row_end=hidden_size,
                col_start=k_offset + h * head_dim,
                col_end=k_offset + (h + 1) * head_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # V — from fused c_attn columns [2H + h*hd : 2H + (h+1)*hd]
        v_offset = 2 * hidden_size
        _add_projection(
            f"L{layer}H{h}_V", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=0, row_end=hidden_size,
                col_start=v_offset + h * head_dim,
                col_end=v_offset + (h + 1) * head_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # O — from c_proj rows [h*hd : (h+1)*hd]
        _add_projection(
            f"L{layer}H{h}_O", {comp},
            ProjectionSlice(
                param_name=out_param,
                row_start=h * head_dim, row_end=(h + 1) * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )


def _build_gpt2_mlp(
    layer: int,
    hidden_size: int,
    intermediate_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build MLP projections for GPT-2 (c_fc + c_proj, no gate)."""
    comp = f"L{layer}_MLP"

    # c_fc (up): [hidden_size, intermediate_size]
    _add_projection(
        f"L{layer}_MLP_up", {comp},
        ProjectionSlice(
            param_name=_gpt2_mlp_fc_param(layer),
            row_start=0, row_end=hidden_size,
            col_start=0, col_end=intermediate_size,
        ),
        projections, proj_to_components, component_to_projs, proj_slices,
    )

    # c_proj (down): [intermediate_size, hidden_size]
    _add_projection(
        f"L{layer}_MLP_down", {comp},
        ProjectionSlice(
            param_name=_gpt2_mlp_proj_param(layer),
            row_start=0, row_end=intermediate_size,
            col_start=0, col_end=hidden_size,
        ),
        projections, proj_to_components, component_to_projs, proj_slices,
    )


def _build_pythia_attention(
    layer: int,
    n_heads: int,
    head_dim: int,
    hidden_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build attention projections for Pythia/GPT-NeoX (fused query_key_value).

    GPT-NeoX interleaves QKV per-head in rows:
      [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
    Each block is head_dim rows. Total = 3 * n_heads * head_dim = 3 * hidden_size.
    """
    fused_param = _pythia_qkv_param(layer)
    out_param = _pythia_attn_out_param(layer)

    for h in range(n_heads):
        comp = f"L{layer}H{h}"
        base = h * 3 * head_dim

        # Q — rows [h*3*hd : h*3*hd + hd]
        _add_projection(
            f"L{layer}H{h}_Q", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=base, row_end=base + head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # K — rows [h*3*hd + hd : h*3*hd + 2*hd]
        _add_projection(
            f"L{layer}H{h}_K", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=base + head_dim, row_end=base + 2 * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # V — rows [h*3*hd + 2*hd : h*3*hd + 3*hd]
        _add_projection(
            f"L{layer}H{h}_V", {comp},
            ProjectionSlice(
                param_name=fused_param,
                row_start=base + 2 * head_dim, row_end=base + 3 * head_dim,
                col_start=0, col_end=hidden_size,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )

        # O — dense columns [h*hd : (h+1)*hd] (input from head h's value output)
        _add_projection(
            f"L{layer}H{h}_O", {comp},
            ProjectionSlice(
                param_name=out_param,
                row_start=0, row_end=hidden_size,
                col_start=h * head_dim, col_end=(h + 1) * head_dim,
            ),
            projections, proj_to_components, component_to_projs, proj_slices,
        )


def _build_pythia_mlp(
    layer: int,
    hidden_size: int,
    intermediate_size: int,
    projections: List[str],
    proj_to_components: Dict[str, Set[str]],
    component_to_projs: Dict[str, List[str]],
    proj_slices: Dict[str, ProjectionSlice],
) -> None:
    """Build MLP projections for Pythia/GPT-NeoX (dense_h_to_4h + dense_4h_to_h)."""
    comp = f"L{layer}_MLP"

    _add_projection(
        f"L{layer}_MLP_up", {comp},
        ProjectionSlice(
            param_name=_pythia_mlp_up_param(layer),
            row_start=0, row_end=intermediate_size,
            col_start=0, col_end=hidden_size,
        ),
        projections, proj_to_components, component_to_projs, proj_slices,
    )

    _add_projection(
        f"L{layer}_MLP_down", {comp},
        ProjectionSlice(
            param_name=_pythia_mlp_down_param(layer),
            row_start=0, row_end=hidden_size,
            col_start=0, col_end=intermediate_size,
        ),
        projections, proj_to_components, component_to_projs, proj_slices,
    )


# =============================================================================
# Public API
# =============================================================================

def build_model_topology(model) -> ModelTopology:
    """
    Build the projection-to-component topology from a HuggingFace model.

    Scans model config to enumerate all projections (weight matrices) and
    components (attention heads, MLP layers), building bidirectional mappings
    with extraction recipes for each projection.

    Args:
        model: HuggingFace model (AutoModelForCausalLM or similar)

    Returns:
        ModelTopology with all mappings populated
    """
    config = model.config
    arch = _detect_architecture(model)
    return _build_topology_from_config(config, arch)


def _build_topology_from_config(config, arch: str) -> ModelTopology:
    """Shared implementation for topology construction from config + arch."""
    n_layers = _get_n_layers(config)
    n_query_heads = _get_n_query_heads(config)
    n_kv_heads = _get_n_kv_heads(config)
    head_dim = _get_head_dim(config)
    hidden_size = config.hidden_size
    intermediate_size = _get_intermediate_size(config)

    projections: List[str] = []
    components: List[str] = []
    proj_to_components: Dict[str, Set[str]] = {}
    component_to_projs: Dict[str, List[str]] = {}
    proj_slices: Dict[str, ProjectionSlice] = {}

    for layer in range(n_layers):
        for h in range(n_query_heads):
            comp = f"L{layer}H{h}"
            components.append(comp)
            component_to_projs[comp] = []

        mlp_comp = f"L{layer}_MLP"
        components.append(mlp_comp)
        component_to_projs[mlp_comp] = []

        if arch == "llama":
            _build_llama_attention(
                layer, n_query_heads, n_kv_heads, head_dim, hidden_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )
        elif arch == "gpt2":
            _build_gpt2_attention(
                layer, n_query_heads, head_dim, hidden_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )
        elif arch == "pythia":
            _build_pythia_attention(
                layer, n_query_heads, head_dim, hidden_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )

        if arch == "llama":
            _build_llama_mlp(
                layer, hidden_size, intermediate_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )
        elif arch == "gpt2":
            _build_gpt2_mlp(
                layer, hidden_size, intermediate_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )
        elif arch == "pythia":
            _build_pythia_mlp(
                layer, hidden_size, intermediate_size,
                projections, proj_to_components, component_to_projs, proj_slices,
            )

    return ModelTopology(
        projections=projections,
        components=components,
        proj_to_components=proj_to_components,
        component_to_projs=component_to_projs,
        proj_slices=proj_slices,
        n_layers=n_layers,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        architecture=arch,
    )


def get_projection_weight(
    model,
    topology: ModelTopology,
    proj_id: str,
) -> Tensor:
    """
    Extract a projection's weight matrix from the model.

    Handles per-head slicing for attention projections and full-matrix
    extraction for MLP projections.

    Args:
        model: HuggingFace model
        topology: Model topology from build_model_topology()
        proj_id: Projection ID (e.g., "L5H3_Q", "L10_MLP_gate")

    Returns:
        2D weight tensor for the projection
    """
    pslice = topology.proj_slices[proj_id]
    param = _resolve_param(model, pslice.param_name)
    return param[pslice.row_slice, pslice.col_slice]


def get_projection_delta(
    state_dict_trained: Dict[str, Tensor],
    state_dict_base: Dict[str, Tensor],
    topology: ModelTopology,
    proj_id: str,
) -> Tensor:
    """
    Compute ΔW for a projection: W_trained - W_base.

    Args:
        state_dict_trained: Trained model state dict
        state_dict_base: Base model (M₀) state dict
        topology: Model topology
        proj_id: Projection ID

    Returns:
        2D delta tensor
    """
    pslice = topology.proj_slices[proj_id]
    W_trained = state_dict_trained[pslice.param_name][pslice.row_slice, pslice.col_slice]
    W_base = state_dict_base[pslice.param_name][pslice.row_slice, pslice.col_slice]
    return W_trained - W_base


def extract_all_projection_deltas(
    state_dict_trained: Dict[str, Tensor],
    state_dict_base: Dict[str, Tensor],
    topology: ModelTopology,
) -> Dict[str, Tensor]:
    """
    Extract ΔW for every projection in the model.

    Args:
        state_dict_trained: Trained model state dict
        state_dict_base: Base model (M₀) state dict
        topology: Model topology

    Returns:
        Dict mapping projection ID to delta tensor
    """
    return {
        proj_id: get_projection_delta(
            state_dict_trained, state_dict_base, topology, proj_id,
        )
        for proj_id in topology.projections
    }


def slice_projection_from_delta(
    delta_dict: Dict[str, Tensor],
    topology: ModelTopology,
    proj_id: str,
) -> Tensor:
    """Slice a projection's delta from a parameter-level delta dict.

    Unlike get_projection_delta() which takes trained + base state dicts and
    computes the difference, this takes an ALREADY-COMPUTED delta dict
    (as returned by DeltaStore.load_delta or trainer.compute_delta).

    Args:
        delta_dict: {param_name: ΔW tensor} — already W_trained - W_base
        topology: Model topology
        proj_id: Projection ID (e.g., "L5H3_Q")

    Returns:
        2D delta tensor for the projection
    """
    pslice = topology.proj_slices[proj_id]
    return delta_dict[pslice.param_name][pslice.row_slice, pslice.col_slice]


def expand_deltas_to_projections(
    delta_dict: Dict[str, Tensor],
    topology: ModelTopology,
) -> Dict[str, Tensor]:
    """Expand a parameter-level delta dict to projection-level.

    Args:
        delta_dict: {param_name: ΔW tensor}
        topology: Model topology

    Returns:
        {proj_id: ΔW tensor} for every projection in the topology
    """
    result = {}
    for proj_id in topology.projections:
        try:
            result[proj_id] = slice_projection_from_delta(delta_dict, topology, proj_id)
        except KeyError:
            continue
    return result


def get_base_weight(
    state_dict_base: Dict[str, Tensor],
    topology: ModelTopology,
    proj_id: str,
) -> Tensor:
    """
    Extract a projection's base weight (W₀) from a state dict.

    Args:
        state_dict_base: Base model (M₀) state dict
        topology: Model topology
        proj_id: Projection ID

    Returns:
        2D base weight tensor
    """
    pslice = topology.proj_slices[proj_id]
    return state_dict_base[pslice.param_name][pslice.row_slice, pslice.col_slice]


# =============================================================================
# Helpers
# =============================================================================

def _resolve_param(model, param_name: str) -> Tensor:
    """Get a parameter tensor from a model by dotted name."""
    parts = param_name.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj.data


def proj_suffix(proj_id: str) -> str:
    """
    Extract the projection suffix from a full projection ID.

    "L5H3_Q" -> "Q"
    "L10_MLP_gate" -> "gate"
    "L3KV1_K" -> "K"
    """
    parts = proj_id.split("_")
    return parts[-1]


def proj_to_component_id(proj_id: str) -> str:
    """
    Extract the component ID from a projection ID.

    "L5H3_Q" -> "L5H3"
    "L10_MLP_gate" -> "L10_MLP"
    "L3KV1_K" -> component depends on GQA mapping (use topology instead)
    """
    if "_MLP_" in proj_id:
        return proj_id.rsplit("_", 1)[0]  # "L10_MLP_gate" -> "L10_MLP"
    elif "KV" in proj_id:
        raise ValueError(
            f"Cannot derive component from shared KV projection {proj_id}. "
            "Use topology.proj_to_components instead."
        )
    else:
        return proj_id.rsplit("_", 1)[0]  # "L5H3_Q" -> "L5H3"


__all__ = [
    "ProjectionSlice",
    "ModelTopology",
    "build_model_topology",
    "get_projection_weight",
    "get_projection_delta",
    "extract_all_projection_deltas",
    "get_base_weight",
    "proj_suffix",
    "proj_to_component_id",
]
