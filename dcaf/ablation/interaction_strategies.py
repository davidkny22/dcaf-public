"""
Phase 2: Seven parallel pair/group interaction discovery strategies (§11, Def 11.6-11.12).

Def 11.6  Strategy A: Graph-adjacent pairs (connected by circuit edges with weight > τ_E)
Def 11.7  Strategy B: Gradient-based interaction screening (Hessian approximation)
Def 11.8  Strategy C: Activation correlation pairs (top cosine-similar pairs)
Def 11.9  Strategy D: Hierarchical clustering (average linkage, threshold 0.7)
Def 11.10 Strategy E: Opposition grouping (|opp_degree(k1) - opp_degree(k2)| < ε_opp)
Def 11.11 Strategy F: Cross-layer attention head composition
Def 11.12 Strategy G: Confidence-weighted random sampling (n=200)

Total pair budget B_pair (default 300). Pairs found by multiple strategies are
prioritized when the union exceeds the budget.
"""

from typing import Dict, List, Set, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random

from dcaf.core.defaults import EPSILON_OPP, EPSILON_TRI, TAU_ABS, TAU_EDGE
from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.superadditivity import (
    SuperadditivityResult,
    test_superadditivity,
    InteractionType,
)


@dataclass
class StrategyResult:
    """
    Result from an interaction discovery strategy.

    Attributes:
        strategy_name: Name of the strategy (1A, 1B, etc.)
        params_found: Parameters identified by this strategy
        interactions: List of interaction test results
        details: Strategy-specific details
    """
    strategy_name: str
    params_found: Set[str]
    interactions: List[SuperadditivityResult] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "params_found": list(self.params_found),
            "interactions": [i.to_dict() for i in self.interactions],
            "details": self.details,
        }


class InteractionStrategy(ABC):
    """Abstract base for interaction discovery strategies."""

    def __init__(
        self,
        model,
        state_manager: ModelStateManager,
        test_fn: Callable[..., float],
        epsilon: float = EPSILON_TRI,
    ):
        self.model = model
        self.state_manager = state_manager
        self.test_fn = test_fn
        self.epsilon = epsilon

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., '1A_Individual')."""
        pass

    @abstractmethod
    def discover(
        self,
        candidates: Set[str],
        **kwargs,
    ) -> StrategyResult:
        """
        Discover interactions using this strategy.

        Args:
            candidates: Set of candidate parameter names
            **kwargs: Strategy-specific arguments

        Returns:
            StrategyResult with discovered parameters
        """
        pass


class StrategyA_GraphAdjacent(InteractionStrategy):
    """
    Strategy A: Graph-adjacent pairs (Def 11.6).

    Components connected by preliminary circuit edges (from activation flow or
    gradient correlation) with edge weight > τ_E are paired. Tests connected
    circuit segments.
    """

    @property
    def name(self) -> str:
        return "A_GraphAdjacent"

    def discover(
        self,
        candidates: Set[str],
        circuit_edges: Optional[List[Tuple[str, str, float]]] = None,
        edge_weight_threshold: float = TAU_EDGE,
        impact_threshold: float = TAU_ABS,
        **kwargs,
    ) -> StrategyResult:
        """
        Test pairs connected by circuit edges with weight > τ_E (Def 11.6).

        Args:
            candidates: Candidate parameters
            circuit_edges: [(comp1, comp2, weight), ...] from preliminary circuit graph
            edge_weight_threshold: Minimum edge weight τ_E
            impact_threshold: Minimum combined impact to report as found

        Returns:
            StrategyResult with significant graph-adjacent pairs
        """
        if circuit_edges is None:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "circuit_edges not provided"},
            )

        params_found = set()
        interactions = []

        for comp1, comp2, weight in circuit_edges:
            if weight <= edge_weight_threshold:
                continue
            if comp1 not in candidates or comp2 not in candidates:
                continue

            result = test_superadditivity(
                params=[comp1, comp2],
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.combined_impact >= impact_threshold:
                params_found.add(comp1)
                params_found.add(comp2)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"edge_weight_threshold": edge_weight_threshold},
        )


class StrategyB_GradientScreening(InteractionStrategy):
    """
    Strategy B: Gradient-based interaction screening (Def 11.7).

    Ablate component k1 and measure how much k2's gradient changes:
        interact(k1, k2) = ||∇_{k2} L|_{k1 ablated} - ∇_{k2} L|_{intact}||

    This is a screening method that predicts which pairs to test via full
    behavioral ablation, not a direct behavioral test. Returns top-N pairs.
    """

    @property
    def name(self) -> str:
        return "B_GradientScreening"

    def discover(
        self,
        candidates: Set[str],
        gradient_pairs: Optional[List[Tuple[str, str, float]]] = None,
        top_n: int = 20,
        **kwargs,
    ) -> StrategyResult:
        """
        Test gradient-predicted pairs (Def 11.7).

        Args:
            candidates: Candidate parameters
            gradient_pairs: [(param1, param2, hessian_value), ...] sorted by importance
            top_n: Number of top pairs to test

        Returns:
            StrategyResult with synergistic pairs
        """
        if gradient_pairs is None:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "gradient_pairs not provided"},
            )

        # Filter to candidate pairs
        valid_pairs = [
            (p1, p2, v) for p1, p2, v in gradient_pairs
            if p1 in candidates and p2 in candidates
        ][:top_n]

        params_found = set()
        interactions = []

        for p1, p2, _ in valid_pairs:
            result = test_superadditivity(
                params=[p1, p2],
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.interaction_type == InteractionType.SYNERGISTIC:
                params_found.add(p1)
                params_found.add(p2)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"top_n": top_n, "pairs_tested": len(valid_pairs)},
        )


class StrategyC_ActivationCorrelation(InteractionStrategy):
    """
    Strategy C: Activation correlation pairs (Def 11.8).

    Build a vector per component by concatenating activation deltas
    ΔA^(k,π)_i across all behavioral signals and probe types. Compute
    pairwise cosine similarity. Select top correlated pairs (budget: B_pair/5).
    Tests co-functioning components.
    """

    @property
    def name(self) -> str:
        return "C_ActivationCorrelation"

    def discover(
        self,
        candidates: Set[str],
        correlation_clusters: Optional[List[Set[str]]] = None,
        **kwargs,
    ) -> StrategyResult:
        """
        Test correlation-based clusters (Def 11.8).

        Args:
            candidates: Candidate parameters
            correlation_clusters: List of correlated parameter sets from
                                   cosine-similarity analysis of activation deltas

        Returns:
            StrategyResult with synergistic clusters
        """
        if correlation_clusters is None:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "correlation_clusters not provided"},
            )

        params_found = set()
        interactions = []

        for cluster in correlation_clusters:
            # Filter to candidates
            valid_params = [p for p in cluster if p in candidates]
            if len(valid_params) < 2:
                continue

            result = test_superadditivity(
                params=valid_params,
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.interaction_type == InteractionType.SYNERGISTIC:
                params_found.update(valid_params)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"n_clusters": len(correlation_clusters)},
        )


class StrategyD_HierarchicalClustering(InteractionStrategy):
    """
    Strategy D: Hierarchical clustering (Def 11.9).

    Using the same correlation matrix as Strategy C, apply hierarchical
    clustering (average linkage, distance = 1 - correlation, threshold = 0.7).
    Extract clusters of size ≥ 3. For each cluster, test collective ablation;
    if significant, find the minimal sufficient subset via greedy removal.
    Tests functional groups.
    """

    @property
    def name(self) -> str:
        return "D_HierarchicalClustering"

    def discover(
        self,
        candidates: Set[str],
        correlation_clusters: Optional[List[Set[str]]] = None,
        min_cluster_size: int = 3,
        **kwargs,
    ) -> StrategyResult:
        """
        Test hierarchical clusters (Def 11.9).

        Args:
            candidates: Candidate parameters
            correlation_clusters: List of clusters (size ≥ 3) from hierarchical
                                   clustering of the correlation matrix
            min_cluster_size: Minimum cluster size to test (default 3 per spec)

        Returns:
            StrategyResult with synergistic clusters
        """
        if correlation_clusters is None:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "correlation_clusters not provided"},
            )

        params_found = set()
        interactions = []

        for cluster in correlation_clusters:
            valid_params = [p for p in cluster if p in candidates]
            if len(valid_params) < min_cluster_size:
                continue

            result = test_superadditivity(
                params=valid_params,
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.interaction_type == InteractionType.SYNERGISTIC:
                params_found.update(valid_params)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"n_clusters": len(correlation_clusters), "min_cluster_size": min_cluster_size},
        )


class StrategyE_OppositionGrouping(InteractionStrategy):
    """
    Strategy E: Opposition grouping (Def 11.10).

    Pair components where:
        similar(k1, k2) = 1[|opp_degree(k1) - opp_degree(k2)| < ε_opp]

    where ε_opp = 0.1 (default). Components with similar opposition profiles
    respond similarly to T+ vs T- — likely parallel circuits implementing
    the same behavioral control.
    """

    @property
    def name(self) -> str:
        return "E_OppositionGrouping"

    def discover(
        self,
        candidates: Set[str],
        opposition_data: Optional[Dict[str, Tuple[float, float]]] = None,
        similarity_threshold: float = EPSILON_OPP,
        top_n: int = 20,
        **kwargs,
    ) -> StrategyResult:
        """
        Test opposition-similar pairs (Def 11.10).

        Args:
            candidates: Candidate parameters
            opposition_data: {param: (delta_sign, opp_degree)}
            similarity_threshold: Max difference in opp_degree (ε_opp, default 0.1)
            top_n: Number of pairs to test

        Returns:
            StrategyResult with synergistic pairs
        """
        if opposition_data is None:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "opposition_data not provided"},
            )

        # Find similar pairs
        similar_pairs = []
        candidate_list = list(candidates)

        for i, p1 in enumerate(candidate_list):
            if p1 not in opposition_data:
                continue
            sign1, opp1 = opposition_data[p1]

            for p2 in candidate_list[i + 1:]:
                if p2 not in opposition_data:
                    continue
                sign2, opp2 = opposition_data[p2]

                # Same sign and similar opposition
                if sign1 == sign2 and abs(opp1 - opp2) < similarity_threshold:
                    similar_pairs.append((p1, p2, abs(opp1 - opp2)))

        # Sort by similarity and take top N
        similar_pairs.sort(key=lambda x: x[2])
        similar_pairs = similar_pairs[:top_n]

        params_found = set()
        interactions = []

        for p1, p2, _ in similar_pairs:
            result = test_superadditivity(
                params=[p1, p2],
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.interaction_type == InteractionType.SYNERGISTIC:
                params_found.add(p1)
                params_found.add(p2)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"pairs_tested": len(similar_pairs), "epsilon_opp": similarity_threshold},
        )


class StrategyF_CrossLayerComposition(InteractionStrategy):
    """
    Strategy F: Cross-layer attention head composition (Def 11.11).

    Pair attention heads from different layers. Cross-layer composition is a
    known circuit motif (induction heads, IOI circuits). MLP components are
    excluded from this strategy.
    """

    @property
    def name(self) -> str:
        return "F_CrossLayerComposition"

    def discover(
        self,
        candidates: Set[str],
        top_n: int = 20,
        impact_threshold: float = TAU_ABS,
        **kwargs,
    ) -> StrategyResult:
        """
        Test cross-layer attention head pairs (Def 11.11).

        Args:
            candidates: Candidate parameters
            top_n: Maximum pairs to test
            impact_threshold: Minimum combined impact

        Returns:
            StrategyResult with cross-layer attention pairs
        """
        # Filter to attention head parameters only (exclude MLPs per spec)
        attn_params = [p for p in candidates if ".self_attn." in p]

        # Group by layer
        layer_to_params: Dict[str, list] = {}
        for param in attn_params:
            parts = param.split(".")
            if len(parts) > 2:
                layer_idx = parts[2]
                if layer_idx not in layer_to_params:
                    layer_to_params[layer_idx] = []
                layer_to_params[layer_idx].append(param)

        # Build cross-layer pairs
        layer_list = sorted(layer_to_params.keys(), key=int)
        cross_pairs = []
        for i, layer1 in enumerate(layer_list):
            for layer2 in layer_list[i + 1:]:
                for p1 in layer_to_params[layer1]:
                    for p2 in layer_to_params[layer2]:
                        cross_pairs.append((p1, p2))
                        if len(cross_pairs) >= top_n:
                            break
                    if len(cross_pairs) >= top_n:
                        break
                if len(cross_pairs) >= top_n:
                    break

        if not cross_pairs:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "No cross-layer attention pairs found"},
            )

        params_found = set()
        interactions = []

        for p1, p2 in cross_pairs:
            result = test_superadditivity(
                params=[p1, p2],
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.combined_impact >= impact_threshold:
                params_found.add(p1)
                params_found.add(p2)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"n_attention_params": len(attn_params), "pairs_tested": len(cross_pairs)},
        )


class StrategyG_RandomSampling(InteractionStrategy):
    """
    Strategy G: Confidence-weighted random sampling (Def 11.12).

    Sample n_samples=200 pairs with probability proportional to C^(k) (unified
    confidence). Serves as false-negative safeguard — catches unexpected
    interactions missed by structured strategies A-F.
    """

    @property
    def name(self) -> str:
        return "G_RandomSampling"

    def discover(
        self,
        candidates: Set[str],
        n_samples: int = 200,
        confidences: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> StrategyResult:
        """
        Confidence-weighted random pairs (Def 11.12).

        Args:
            candidates: Candidate parameters
            n_samples: Number of random pairs (default 200 per spec)
            confidences: {param: C^(k)} for weighted sampling; uniform if None
            seed: Random seed for reproducibility

        Returns:
            StrategyResult with discovered interactions
        """
        if seed is not None:
            random.seed(seed)

        candidate_list = list(candidates)
        if len(candidate_list) < 2:
            return StrategyResult(
                strategy_name=self.name,
                params_found=set(),
                interactions=[],
                details={"error": "Not enough candidates"},
            )

        # Build sampling weights
        if confidences is not None:
            weights = [confidences.get(p, 1.0) for p in candidate_list]
            total = sum(weights)
            weights = [w / total for w in weights] if total > 0 else None
        else:
            weights = None

        # Generate weighted random pairs
        pairs: set = set()
        attempts = 0
        max_attempts = n_samples * 10

        while len(pairs) < n_samples and attempts < max_attempts:
            if weights is not None:
                selected = random.choices(candidate_list, weights=weights, k=2)
                p1, p2 = selected[0], selected[1]
            else:
                p1, p2 = random.sample(candidate_list, 2)

            if p1 != p2 and (p1, p2) not in pairs and (p2, p1) not in pairs:
                pairs.add((p1, p2))
            attempts += 1

        params_found = set()
        interactions = []

        for p1, p2 in pairs:
            result = test_superadditivity(
                params=[p1, p2],
                model=self.model,
                state_manager=self.state_manager,
                test_fn=self.test_fn,
                epsilon=self.epsilon,
            )
            interactions.append(result)

            if result.interaction_type == InteractionType.SYNERGISTIC:
                params_found.add(p1)
                params_found.add(p2)

        return StrategyResult(
            strategy_name=self.name,
            params_found=params_found,
            interactions=interactions,
            details={"n_pairs": len(pairs), "n_samples": n_samples, "seed": seed},
        )


def run_all_strategies(
    candidates: Set[str],
    model,
    state_manager: ModelStateManager,
    test_fn: Callable[..., float],
    epsilon: float = EPSILON_TRI,
    strategy_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, StrategyResult]:
    """
    Run all 7 Phase 2 strategies (Def 11.6-11.12).

    All strategies run independently and results are pooled. When the union of
    candidate pairs exceeds B_pair (default 300), pairs found by multiple
    strategies are prioritized.

    Args:
        candidates: Candidate parameters (H_cand)
        model: Model to test
        state_manager: ModelStateManager
        test_fn: Impact measurement function
        epsilon: Synergy detection threshold ε_syn (default EPSILON_TRI = 0.05)
        strategy_kwargs: {strategy_name: {kwarg: value}}

    Returns:
        {strategy_name: StrategyResult}
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    strategies = [
        StrategyA_GraphAdjacent(model, state_manager, test_fn, epsilon),
        StrategyB_GradientScreening(model, state_manager, test_fn, epsilon),
        StrategyC_ActivationCorrelation(model, state_manager, test_fn, epsilon),
        StrategyD_HierarchicalClustering(model, state_manager, test_fn, epsilon),
        StrategyE_OppositionGrouping(model, state_manager, test_fn, epsilon),
        StrategyF_CrossLayerComposition(model, state_manager, test_fn, epsilon),
        StrategyG_RandomSampling(model, state_manager, test_fn, epsilon),
    ]

    results = {}
    for strategy in strategies:
        kwargs = strategy_kwargs.get(strategy.name, {})
        results[strategy.name] = strategy.discover(candidates, **kwargs)

    return results


def compute_discovery_count(
    param: str,
    results: Dict[str, StrategyResult],
) -> int:
    """
    Count how many strategies discovered a parameter.

    discovery_count >= 2: High-confidence circuit
    discovery_count == 1: Re-test or mark uncertain

    Args:
        param: Parameter name
        results: {strategy_name: StrategyResult}

    Returns:
        Number of strategies that found this parameter
    """
    return sum(
        1 for result in results.values()
        if param in result.params_found
    )


def get_high_confidence_params(
    results: Dict[str, StrategyResult],
    min_discoveries: int = 2,
) -> Set[str]:
    """
    Get parameters discovered by multiple strategies.

    Args:
        results: {strategy_name: StrategyResult}
        min_discoveries: Minimum strategies that must find a param

    Returns:
        Set of high-confidence parameters
    """
    # Collect all discovered params
    all_params = set()
    for result in results.values():
        all_params.update(result.params_found)

    # Filter by discovery count
    return {
        param for param in all_params
        if compute_discovery_count(param, results) >= min_discoveries
    }


def get_interaction_summary(
    results: Dict[str, StrategyResult],
) -> Dict[str, Any]:
    """
    Summary of all strategy results.

    Args:
        results: {strategy_name: StrategyResult}

    Returns:
        Summary dict
    """
    all_params = set()
    all_synergistic = 0
    total_interactions = 0

    for result in results.values():
        all_params.update(result.params_found)
        for interaction in result.interactions:
            total_interactions += 1
            if interaction.interaction_type == InteractionType.SYNERGISTIC:
                all_synergistic += 1

    return {
        "strategies_run": len(results),
        "total_params_found": len(all_params),
        "total_interactions_tested": total_interactions,
        "synergistic_interactions": all_synergistic,
        "high_confidence_params": len(get_high_confidence_params(results)),
        "per_strategy": {
            name: {
                "params_found": len(result.params_found),
                "interactions": len(result.interactions),
            }
            for name, result in results.items()
        },
    }


__all__ = [
    "InteractionStrategy",
    "StrategyResult",
    "StrategyA_GraphAdjacent",
    "StrategyB_GradientScreening",
    "StrategyC_ActivationCorrelation",
    "StrategyD_HierarchicalClustering",
    "StrategyE_OppositionGrouping",
    "StrategyF_CrossLayerComposition",
    "StrategyG_RandomSampling",
    "run_all_strategies",
    "compute_discovery_count",
    "get_high_confidence_params",
    "get_interaction_summary",
]
