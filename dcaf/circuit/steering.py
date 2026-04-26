"""
Steering vector optimization (§10, Def 10.1-10.5).

Implements bidirectional steering vector optimization and analysis:

- Def 10.1: Target steering vector v*_{k,+} = argmax_v L_target(M0 + v·e_k)
- Def 10.2: Opposite steering vector v*_{k,-} = argmax_v L_opposite(M0 + v·e_k)
- Def 10.3: Steering-geometry alignment metrics (alpha_+, alpha_-, alpha_pm)
- Def 10.4: Steering effectiveness measurement
- Def 10.5: Defensive vectors for protection (reinforce / block_attack)

Note: This is the real §10 implementation. The module dcaf.steering.vectors
is a backward-compatible stub that re-exports from here.
"""

from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn.functional as F

from dcaf.core.defaults import EPS_GENERAL


@dataclass
class SteeringVector:
    """
    Optimized steering vector for a component.

    Attributes:
        component: Component ID
        v_plus: Steering vector toward target behavior
        v_minus: Steering vector toward opposite behavior
        eff_plus: Effectiveness of v_plus
        eff_minus: Effectiveness of v_minus
        optimization_steps: Number of optimization steps used
    """
    component: str
    v_plus: Tensor
    v_minus: Tensor
    eff_plus: float
    eff_minus: float
    optimization_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "v_plus_norm": float(torch.norm(self.v_plus).item()),
            "v_minus_norm": float(torch.norm(self.v_minus).item()),
            "eff_plus": self.eff_plus,
            "eff_minus": self.eff_minus,
            "optimization_steps": self.optimization_steps,
        }


@dataclass
class SteeringAlignment:
    """
    Alignment between steering vectors and geometric direction.

    Attributes:
        alpha_plus: cos(v_plus, d) - target steering vs representation
        alpha_minus: cos(v_minus, d) - opposite steering vs representation
        alpha_opposition: cos(v_plus, v_minus) - are they actually opposites?
    """
    alpha_plus: float
    alpha_minus: float
    alpha_opposition: float

    @property
    def is_clean_control(self) -> bool:
        """
        Check if this is clean single-axis control.

        Pattern: α_+ ≈ 1, α_- ≈ -1, α_± ≈ -1
        Meaning: Representation = mechanism
        """
        return (
            self.alpha_plus > 0.8 and
            self.alpha_minus < -0.8 and
            self.alpha_opposition < -0.8
        )

    @property
    def is_orthogonal_pathways(self) -> bool:
        """
        Check if target and opposite use orthogonal pathways.

        Pattern: α_± ≈ 0
        Meaning: Different mechanisms for target vs opposite
        """
        return abs(self.alpha_opposition) < 0.3

    @property
    def is_anomalous(self) -> bool:
        """
        Check for anomalous alignment (both push same direction).

        Pattern: α_± > 0
        Meaning: Unexpected - investigate further
        """
        return self.alpha_opposition > 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_plus": self.alpha_plus,
            "alpha_minus": self.alpha_minus,
            "alpha_opposition": self.alpha_opposition,
            "is_clean_control": self.is_clean_control,
            "is_orthogonal_pathways": self.is_orthogonal_pathways,
            "is_anomalous": self.is_anomalous,
        }


@dataclass
class SteeringAnalysis:
    """
    Complete steering analysis for a component.

    Attributes:
        component: Component ID
        steering_vector: Optimized steering vectors
        alignment: Alignment with geometric direction
        defensive_vectors: Attack-blocking vectors
    """
    component: str
    steering_vector: SteeringVector
    alignment: Optional[SteeringAlignment] = None
    defensive_vectors: Dict[str, Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "steering_vector": self.steering_vector.to_dict(),
            "alignment": self.alignment.to_dict() if self.alignment else None,
            "has_defensive_vectors": len(self.defensive_vectors) > 0,
        }


def compute_cosine_similarity(
    v1: Tensor,
    v2: Tensor,
    eps: float = EPS_GENERAL,
) -> float:
    """
    Compute cosine similarity between vectors.

    Args:
        v1: First vector
        v2: Second vector
        eps: Numerical stability

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    if norm1 < eps or norm2 < eps:
        return 0.0

    return F.cosine_similarity(
        v1.flatten().unsqueeze(0),
        v2.flatten().unsqueeze(0)
    ).item()


def optimize_steering_vector(
    model,
    component: str,
    L_behav: Callable[..., Tensor],
    component_dim: int,
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
    lr: float = 0.01,
    steps: int = 100,
    device: str = "cuda",
) -> Tensor:
    """
    Optimize steering vector for a component.

    v*_k = argmax_v L_behav(M0 + v·e_k)

    Args:
        model: Model to steer
        component: Component ID
        L_behav: Behavioral loss function (higher = more target behavior)
        component_dim: Dimension of component activations
        inject_fn: Function to inject steering vector
        clear_fn: Function to clear steering
        lr: Learning rate
        steps: Optimization steps
        device: Device for tensors

    Returns:
        Optimized steering vector
    """
    v = torch.zeros(component_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        # Inject steering
        inject_fn(component, v)

        # Compute loss (negate because we maximize)
        loss = -L_behav(model)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Clear steering for next iteration
        clear_fn(component)

    return v.detach()


def optimize_bidirectional_steering(
    model,
    component: str,
    L_target: Callable[..., Tensor],
    L_opposite: Callable[..., Tensor],
    component_dim: int,
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
    lr: float = 0.01,
    steps: int = 100,
    device: str = "cuda",
) -> SteeringVector:
    """
    Optimize steering vectors in both directions.

    v*_{k,+} = argmax_v L_target(M0 + v·e_k)
    v*_{k,-} = argmax_v L_opposite(M0 + v·e_k)

    Args:
        model: Model to steer
        component: Component ID
        L_target: Target behavior loss
        L_opposite: Opposite behavior loss
        component_dim: Dimension of component activations
        inject_fn: Function to inject steering vector
        clear_fn: Function to clear steering
        lr: Learning rate
        steps: Optimization steps
        device: Device for tensors

    Returns:
        SteeringVector with v_plus and v_minus
    """
    # Optimize toward target
    v_plus = optimize_steering_vector(
        model, component, L_target, component_dim,
        inject_fn, clear_fn, lr, steps, device
    )

    # Optimize toward opposite
    v_minus = optimize_steering_vector(
        model, component, L_opposite, component_dim,
        inject_fn, clear_fn, lr, steps, device
    )

    return SteeringVector(
        component=component,
        v_plus=v_plus,
        v_minus=v_minus,
        eff_plus=0.0,  # Computed separately
        eff_minus=0.0,
        optimization_steps=steps,
    )


def compute_steering_effectiveness(
    model,
    component: str,
    v: Tensor,
    behavior_metric: Callable[..., float],
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
) -> float:
    """
    Measure behavioral shift from steering.

    eff = |behavior_metric(with_steering) - behavior_metric(without_steering)|

    Args:
        model: Model to steer
        component: Component ID
        v: Steering vector
        behavior_metric: Function returning behavioral score
        inject_fn: Function to inject steering vector
        clear_fn: Function to clear steering

    Returns:
        Effectiveness (magnitude of behavioral shift)
    """
    # Baseline without steering
    score_without = behavior_metric(model)

    # With steering
    inject_fn(component, v)
    score_with = behavior_metric(model)
    clear_fn(component)

    return abs(score_with - score_without)


def compute_bidirectional_effectiveness(
    model,
    component: str,
    v_plus: Tensor,
    v_minus: Tensor,
    target_metric: Callable[..., float],
    opposite_metric: Callable[..., float],
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
) -> Tuple[float, float]:
    """
    Compute effectiveness for both steering directions.

    Args:
        model: Model to steer
        component: Component ID
        v_plus: Target steering vector
        v_minus: Opposite steering vector
        target_metric: Target behavior metric
        opposite_metric: Opposite behavior metric
        inject_fn: Injection function
        clear_fn: Clear function

    Returns:
        (eff_plus, eff_minus)
    """
    eff_plus = compute_steering_effectiveness(
        model, component, v_plus, target_metric, inject_fn, clear_fn
    )
    eff_minus = compute_steering_effectiveness(
        model, component, v_minus, opposite_metric, inject_fn, clear_fn
    )

    return eff_plus, eff_minus


def compute_steering_alignment(
    v_plus: Tensor,
    v_minus: Tensor,
    d: Tensor,
) -> SteeringAlignment:
    """
    Compute alignment between steering vectors and geometric direction.

    α_+ = cos(v_plus, d)
    α_- = cos(v_minus, d)
    α_± = cos(v_plus, v_minus)

    Args:
        v_plus: Target steering vector
        v_minus: Opposite steering vector
        d: Geometric contrastive direction

    Returns:
        SteeringAlignment
    """
    alpha_plus = compute_cosine_similarity(v_plus, d)
    alpha_minus = compute_cosine_similarity(v_minus, d)
    alpha_opposition = compute_cosine_similarity(v_plus, v_minus)

    return SteeringAlignment(
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        alpha_opposition=alpha_opposition,
    )


def get_defensive_vectors(
    v_plus: Tensor,
    v_minus: Tensor,
) -> Dict[str, Tensor]:
    """
    Get defensive vectors for protection.

    When α_± ≠ -1 (vectors aren't opposites):
    - v_plus = "Reinforce target behavior" (defensive)
    - -v_minus = "Block attack pathway" (directly counters vulnerability)

    For protection, -v_minus may be more robust because it
    directly counters the actual attack vector.

    Args:
        v_plus: Target steering vector
        v_minus: Opposite steering vector

    Returns:
        {
            "reinforce": v_plus,
            "block_attack": -v_minus
        }
    """
    return {
        "reinforce": v_plus,
        "block_attack": -v_minus,
    }


def compute_full_steering_analysis(
    model,
    component: str,
    L_target: Callable[..., Tensor],
    L_opposite: Callable[..., Tensor],
    target_metric: Callable[..., float],
    opposite_metric: Callable[..., float],
    d: Tensor,
    component_dim: int,
    inject_fn: Callable[[str, Tensor], None],
    clear_fn: Callable[[str], None],
    lr: float = 0.01,
    steps: int = 100,
    device: str = "cuda",
) -> SteeringAnalysis:
    """
    Complete steering analysis for a component.

    Args:
        model: Model to steer
        component: Component ID
        L_target: Target behavior loss
        L_opposite: Opposite behavior loss
        target_metric: Target behavior metric
        opposite_metric: Opposite behavior metric
        d: Geometric contrastive direction
        component_dim: Component activation dimension
        inject_fn: Injection function
        clear_fn: Clear function
        lr: Learning rate
        steps: Optimization steps
        device: Device for tensors

    Returns:
        SteeringAnalysis with vectors, alignment, and defensive vectors
    """
    # Optimize steering vectors
    steering_vector = optimize_bidirectional_steering(
        model, component, L_target, L_opposite,
        component_dim, inject_fn, clear_fn, lr, steps, device
    )

    # Compute effectiveness
    eff_plus, eff_minus = compute_bidirectional_effectiveness(
        model, component, steering_vector.v_plus, steering_vector.v_minus,
        target_metric, opposite_metric, inject_fn, clear_fn
    )
    steering_vector.eff_plus = eff_plus
    steering_vector.eff_minus = eff_minus

    # Compute alignment
    alignment = compute_steering_alignment(
        steering_vector.v_plus,
        steering_vector.v_minus,
        d,
    )

    # Get defensive vectors
    defensive_vectors = get_defensive_vectors(
        steering_vector.v_plus,
        steering_vector.v_minus,
    )

    return SteeringAnalysis(
        component=component,
        steering_vector=steering_vector,
        alignment=alignment,
        defensive_vectors=defensive_vectors,
    )


def rank_by_effectiveness(
    analyses: Dict[str, SteeringAnalysis],
    direction: str = "plus",
) -> List[Tuple[str, float]]:
    """
    Rank components by steering effectiveness.

    Args:
        analyses: {component: SteeringAnalysis}
        direction: "plus" or "minus"

    Returns:
        [(component, effectiveness), ...] sorted descending
    """
    if direction == "plus":
        items = [(c, a.steering_vector.eff_plus) for c, a in analyses.items()]
    else:
        items = [(c, a.steering_vector.eff_minus) for c, a in analyses.items()]

    return sorted(items, key=lambda x: x[1], reverse=True)


def get_steering_summary(
    analyses: Dict[str, SteeringAnalysis],
) -> Dict[str, Any]:
    """
    Summary statistics for steering analyses.

    Args:
        analyses: {component: SteeringAnalysis}

    Returns:
        Summary dict
    """
    if not analyses:
        return {"count": 0}

    eff_plus = [a.steering_vector.eff_plus for a in analyses.values()]
    eff_minus = [a.steering_vector.eff_minus for a in analyses.values()]

    clean_control = sum(1 for a in analyses.values()
                        if a.alignment and a.alignment.is_clean_control)
    orthogonal = sum(1 for a in analyses.values()
                     if a.alignment and a.alignment.is_orthogonal_pathways)
    anomalous = sum(1 for a in analyses.values()
                    if a.alignment and a.alignment.is_anomalous)

    return {
        "count": len(analyses),
        "mean_eff_plus": sum(eff_plus) / len(eff_plus),
        "mean_eff_minus": sum(eff_minus) / len(eff_minus),
        "max_eff_plus": max(eff_plus),
        "max_eff_minus": max(eff_minus),
        "clean_control_count": clean_control,
        "orthogonal_pathways_count": orthogonal,
        "anomalous_count": anomalous,
    }


__all__ = [
    "SteeringVector",
    "SteeringAlignment",
    "SteeringAnalysis",
    "compute_cosine_similarity",
    "optimize_steering_vector",
    "optimize_bidirectional_steering",
    "compute_steering_effectiveness",
    "compute_bidirectional_effectiveness",
    "compute_steering_alignment",
    "get_defensive_vectors",
    "compute_full_steering_analysis",
    "rank_by_effectiveness",
    "get_steering_summary",
]
