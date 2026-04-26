"""
Edge discovery methods for circuit graph (§9).

Implements the five edge discovery methods from §9:

- Method 1: Activation Flow Analysis — corr(A_k, A_k') across layers
- Method 2: Targeted Ablation — E^abl_{k→k'} = ||A^(k')_with_k - A^(k')_without_k||_F
- Method 3: Targeted Steering — E^steer_{k→k'} = ||A^(k')_with_v - A^(k')_without_v||_F
- Method 4: Gradient Flow Analysis — grad_corr(k, k') = Corr(grad_W_k L, grad_W_k' L)
- Method 5: Cross-Layer Attention Patterns — attn_flow(k_l, k_l')

Combined edge weight: E_{k→k'} = max(E^abl, E^steer)  [Causal Edge definition]
Edge predicate: e(k, k') = 1[E_{k→k'} >= tau_E AND k' in V]
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
from torch import Tensor

from dcaf.core.defaults import TAU_EDGE, EPS_GENERAL


class EdgeMethod(Enum):
    """Method used to discover an edge."""
    ABLATION = "ablation"
    STEERING = "steering"
    GRADIENT = "gradient"
    ATTENTION = "attention"
    CORRELATION = "correlation"


@dataclass
class EdgeWeight:
    """
    Discovered edge with weight and metadata.

    Attributes:
        source: Source component ID
        target: Target component ID
        weight: Edge weight (0.0 to 1.0)
        method: Discovery method used
        details: Additional information
    """
    source: str
    target: str
    weight: float
    method: EdgeMethod
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "method": self.method.value,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeWeight":
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data["weight"],
            method=EdgeMethod(data["method"]),
            details=data.get("details", {}),
        )


def edge_activation_flow(
    A_k: Tensor,
    A_k_prime: Tensor,
) -> float:
    """
    Compute activation flow correlation between components.

    corr(A_i^(k), A_i^(k'))

    Args:
        A_k: Activations at component k [n_samples, d_k]
        A_k_prime: Activations at component k' [n_samples, d_k']

    Returns:
        Correlation coefficient in [-1, 1]
    """
    # Flatten to vectors
    a_k = A_k.flatten().float()
    a_k_prime = A_k_prime.flatten().float()

    # Compute correlation
    mean_k = a_k.mean()
    mean_k_prime = a_k_prime.mean()

    centered_k = a_k - mean_k
    centered_k_prime = a_k_prime - mean_k_prime

    cov = (centered_k * centered_k_prime).mean()
    std_k = centered_k.std() + EPS_GENERAL
    std_k_prime = centered_k_prime.std() + EPS_GENERAL

    return (cov / (std_k * std_k_prime)).item()


def edge_ablation(
    get_activations_fn: Callable[[str], Tensor],
    ablate_fn: Callable[[str], None],
    restore_fn: Callable[[str], None],
    k: str,
    k_prime: str,
) -> float:
    """
    Compute ablation-based edge weight.

    E^abl_{k→k'} = ||A^(k')_with_k - A^(k')_without_k||_F

    Args:
        get_activations_fn: Function to get activations at a component
        ablate_fn: Function to ablate a component
        restore_fn: Function to restore a component
        k: Source component
        k_prime: Target component

    Returns:
        Frobenius norm of activation difference
    """
    # Get activations with k intact
    A_with = get_activations_fn(k_prime)

    # Ablate k and get activations at k'
    ablate_fn(k)
    A_without = get_activations_fn(k_prime)
    restore_fn(k)

    # Compute Frobenius norm of difference
    diff = A_with - A_without
    return torch.norm(diff, p='fro').item()


def edge_steering(
    get_activations_fn: Callable[[str], Tensor],
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
    k: str,
    k_prime: str,
    v_k: Tensor,
) -> float:
    """
    Compute steering-based edge weight.

    E^steer_{k→k'} = ||A^(k')_with_v - A^(k')_without_v||_F

    Args:
        get_activations_fn: Function to get activations at a component
        inject_fn: Function to inject steering vector at a component
        clear_fn: Function to clear steering injection
        k: Source component
        k_prime: Target component
        v_k: Steering vector for component k

    Returns:
        Frobenius norm of activation difference
    """
    # Get activations without steering
    A_without = get_activations_fn(k_prime)

    # Inject steering and get activations
    inject_fn(k, v_k)
    A_with = get_activations_fn(k_prime)
    clear_fn(k)

    # Compute Frobenius norm of difference
    diff = A_with - A_without
    return torch.norm(diff, p='fro').item()


def edge_gradient_flow(
    gradients_k: Tensor,
    gradients_k_prime: Tensor,
) -> float:
    """
    Compute gradient flow correlation.

    grad_corr(k, k') = Corr(∇_{W_k}L, ∇_{W_k'}L)

    Args:
        gradients_k: Gradients at component k
        gradients_k_prime: Gradients at component k'

    Returns:
        Correlation coefficient
    """
    return edge_activation_flow(gradients_k, gradients_k_prime)


def compute_edge_weight(
    E_abl: float,
    E_steer: float,
) -> float:
    """
    Compute combined edge weight.

    E_{k→k'} = max(E^abl, E^steer)

    Args:
        E_abl: Ablation-based weight
        E_steer: Steering-based weight

    Returns:
        Combined edge weight
    """
    return max(E_abl, E_steer)


def normalize_edge_weights(
    edges: List[EdgeWeight],
    eps: float = EPS_GENERAL,
) -> List[EdgeWeight]:
    """
    Normalize edge weights to [0, 1].

    Args:
        edges: List of edges
        eps: Minimum value to avoid division by zero

    Returns:
        List of edges with normalized weights
    """
    if not edges:
        return []

    max_weight = max(e.weight for e in edges)
    if max_weight < eps:
        return edges

    normalized = []
    for edge in edges:
        normalized.append(EdgeWeight(
            source=edge.source,
            target=edge.target,
            weight=edge.weight / max_weight,
            method=edge.method,
            details=edge.details,
        ))

    return normalized


def discover_edges_correlation(
    activations: Dict[str, Tensor],
    components: List[str],
    threshold: float = TAU_EDGE,
) -> List[EdgeWeight]:
    """
    Discover edges using activation correlation.

    Args:
        activations: {component: activations}
        components: List of component IDs
        threshold: Minimum correlation for edge

    Returns:
        List of discovered edges
    """
    edges = []

    for i, k in enumerate(components):
        if k not in activations:
            continue

        for k_prime in components[i + 1:]:
            if k_prime not in activations:
                continue

            corr = abs(edge_activation_flow(activations[k], activations[k_prime]))

            if corr >= threshold:
                edges.append(EdgeWeight(
                    source=k,
                    target=k_prime,
                    weight=corr,
                    method=EdgeMethod.CORRELATION,
                ))
                # Add reverse edge (correlation is symmetric)
                edges.append(EdgeWeight(
                    source=k_prime,
                    target=k,
                    weight=corr,
                    method=EdgeMethod.CORRELATION,
                ))

    return edges


def discover_edges_ablation(
    components: List[str],
    state_manager,
    get_activations_fn: Callable[[str, Any], Dict[str, Tensor]],
    model,
    threshold: float = TAU_EDGE,
) -> List[EdgeWeight]:
    """
    Discover edges using ablation.

    Args:
        components: List of component IDs
        state_manager: ModelStateManager for ablation
        get_activations_fn: Function to get activations
        model: Model to use
        threshold: Minimum weight for edge

    Returns:
        List of discovered edges
    """
    edges = []

    for k in components:
        k_params = _get_component_params(k, state_manager)
        if not k_params:
            continue

        # Get baseline activations
        state_manager.reset_to_safety()
        baseline = get_activations_fn(model, components)

        # Ablate and measure impact on other components
        with state_manager.temporary_ablation(k_params):
            ablated = get_activations_fn(model, components)

        for k_prime in components:
            if k == k_prime:
                continue
            if k_prime not in baseline or k_prime not in ablated:
                continue

            diff = baseline[k_prime] - ablated[k_prime]
            weight = torch.norm(diff, p='fro').item()

            if weight >= threshold:
                edges.append(EdgeWeight(
                    source=k,
                    target=k_prime,
                    weight=weight,
                    method=EdgeMethod.ABLATION,
                ))

    return normalize_edge_weights(edges)


def _get_component_params(component: str, state_manager) -> List[str]:
    """Get parameters for a component.

    Delegates to the canonical implementation in dcaf.arch.transformer.
    """
    from dcaf.arch.transformer import get_component_params
    return get_component_params(component, state_manager.get_delta_params())


def discover_edges(
    components: List[str],
    model,
    state_manager,
    get_activations_fn: Optional[Callable] = None,
    activations: Optional[Dict[str, Tensor]] = None,
    methods: List[str] = None,
    threshold: float = TAU_EDGE,
) -> List[EdgeWeight]:
    """
    Discover all edges between components.

    Args:
        components: List of component IDs
        model: Model to use
        state_manager: ModelStateManager
        get_activations_fn: Function to get activations (for ablation method)
        activations: Pre-computed activations (for correlation method)
        methods: List of methods to use ["ablation", "correlation"]
        threshold: Minimum weight for edge

    Returns:
        List of all discovered edges
    """
    if methods is None:
        methods = ["correlation"]

    all_edges = []

    if "correlation" in methods and activations is not None:
        edges = discover_edges_correlation(activations, components, threshold)
        all_edges.extend(edges)

    if "ablation" in methods and get_activations_fn is not None:
        edges = discover_edges_ablation(
            components, state_manager, get_activations_fn, model, threshold
        )
        all_edges.extend(edges)

    return all_edges


def merge_edges(
    edges: List[EdgeWeight],
) -> List[EdgeWeight]:
    """
    Merge edges with same source/target, keeping max weight.

    Args:
        edges: List of edges (may have duplicates)

    Returns:
        List of merged edges
    """
    edge_dict = {}

    for edge in edges:
        key = (edge.source, edge.target)
        if key not in edge_dict or edge.weight > edge_dict[key].weight:
            edge_dict[key] = edge

    return list(edge_dict.values())


def filter_edges(
    edges: List[EdgeWeight],
    threshold: float = TAU_EDGE,
) -> List[EdgeWeight]:
    """
    Filter edges by weight threshold.

    Args:
        edges: List of edges
        threshold: Minimum weight

    Returns:
        Filtered list of edges
    """
    return [e for e in edges if e.weight >= threshold]


def get_edge_summary(
    edges: List[EdgeWeight],
) -> Dict[str, Any]:
    """
    Summary statistics for edges.

    Args:
        edges: List of edges

    Returns:
        Summary dict
    """
    if not edges:
        return {"count": 0}

    weights = [e.weight for e in edges]
    method_counts = {}
    for edge in edges:
        m = edge.method.value
        method_counts[m] = method_counts.get(m, 0) + 1

    # Count unique nodes
    sources = set(e.source for e in edges)
    targets = set(e.target for e in edges)
    all_nodes = sources | targets

    return {
        "count": len(edges),
        "unique_nodes": len(all_nodes),
        "mean_weight": sum(weights) / len(weights),
        "max_weight": max(weights),
        "min_weight": min(weights),
        "method_counts": method_counts,
    }


__all__ = [
    "EdgeMethod",
    "EdgeWeight",
    "edge_activation_flow",
    "edge_ablation",
    "edge_steering",
    "edge_gradient_flow",
    "compute_edge_weight",
    "normalize_edge_weights",
    "discover_edges_correlation",
    "discover_edges_ablation",
    "discover_edges",
    "merge_edges",
    "filter_edges",
    "get_edge_summary",
]
