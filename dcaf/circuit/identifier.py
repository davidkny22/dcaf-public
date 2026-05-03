"""
Circuit identification pipeline (sec:circuit-graph; def:behavioral-circuit-graph).

Implements the 7-step circuit identification pipeline:
1. Component Identification — Map weight parameters to components
2. Causal Connectivity via Ablation — Add edges from ablation effects
3. Attention Flow Refinement — Add correlational edges from attention patterns
4. Graph Construction — Build directed graph G = (V, E)
5. Circuit Extraction — Cluster into circuits (3 methods: disjoint, probe-response, functional)
6. Circuit Validation — Test superadditive effects (sec:ablation)
7. Flow Computation — Topological sort respecting layer order

Based on:
- Causal tracing (Meng et al., ROME)
- Activation patching (Anthropic IOI paper)
- Standard ablation studies extended to track cross-component effects
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set

import numpy as np
import torch

from dcaf.circuit.graph import CircuitGraph
from dcaf.circuit.results import (
    Circuit,
    CircuitAnalysisResults,
    CircuitValidation,
)
from dcaf.core.defaults import ATTENTION_WEIGHT, TAU_EDGE
from dcaf.domains.activation import ActivationDelta, ProbeSet

logger = logging.getLogger(__name__)

ClusteringMethod = Literal["disjoint", "probe-response", "functional"]

if TYPE_CHECKING:
    from dcaf.ablation.results import WeightClassification


@dataclass
class _FunctionalClusterLabel:
    """Internal label for functional clustering (harmful vs neutral response).

    Used only within extract_circuits_functional — not part of public API.
    For the public functional classification type, see dcaf.circuit.classification.
    """

    component: str
    harmful_activation: float
    neutral_activation: float
    category: str  # "safety", "capability", "shared", "inactive"


class CircuitIdentifier:
    """
    Identify circuits from DCAF weight candidates and activation analysis.

    Builds a directed graph of component connectivity and clusters it into
    circuits using one of three methods: disjoint, probe-response, or functional.

    **Edge Types:**
    1. Ablation edges (weight=1.0): Causal connections from ablation studies
       - Direction: component A → component B means ablating A affects B
       - Highest confidence (direct causal measurement)

    2. Attention edges (weight=0.3): Correlational connections from attention flow
       - Direction: head A → head B means B reads from A's write positions
       - Lower confidence (correlational pattern)

    **Node Properties:**
    - generation_score: Delta-of-deltas signal indicating generation steering strength
    - is_generation_steering: Boolean flag for components that differentiate safe/unsafe paths
    - Generation steering is a node property (not an edge type) since it's intrinsic to the component
    """

    def __init__(
        self,
        training_delta: ActivationDelta,
        ablation_deltas: Dict[str, ActivationDelta],
        weight_candidates: List[str],
        probe_set: ProbeSet,
        edge_threshold: float = TAU_EDGE,
        attention_weight: float = ATTENTION_WEIGHT,
    ):
        """
        Initialize circuit identifier.

        Args:
            training_delta: Activation delta from training (pre vs post)
            ablation_deltas: Dict mapping weight name to its ablation delta
            weight_candidates: List of DCAF-identified weight names
            probe_set: Probe set used for activation capture
            edge_threshold: Minimum activation change to create edge
            attention_weight: Weight factor for attention edges (< 1 since correlational)
        """
        self.training_delta = training_delta
        self.ablation_deltas = ablation_deltas
        self.weight_candidates = weight_candidates
        self.probe_set = probe_set
        self.edge_threshold = edge_threshold
        self.attention_weight = attention_weight
        self.graph = CircuitGraph()

        # Cache for ablation impacts
        self._ablation_impacts: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Component Identification
    # ─────────────────────────────────────────────────────────────

    def map_weight_to_component(self, weight_name: str) -> str:
        """
        Map weight parameter name to component name.

        Args:
            weight_name: Full parameter name (e.g., "model.layers.10.mlp.down_proj.weight")

        Returns:
            Component name (e.g., "L10_MLP" or "L10H")
        """
        # Extract layer number — handles LLaMA (model.layers.N), GPT-2
        # (transformer.h.N), and Pythia (gpt_neox.layers.N)
        layer_match = re.search(r"(?:layers?|\.h)[._](\d+)", weight_name)
        if not layer_match:
            # Also try projection ID format (L5H3_Q, L10_MLP_gate)
            proj_match = re.match(r"L(\d+)", weight_name)
            if proj_match:
                if "_MLP" in weight_name:
                    return f"L{proj_match.group(1)}_MLP"
                head_match = re.match(r"L\d+H(\d+)", weight_name)
                if head_match:
                    return f"L{proj_match.group(1)}H{head_match.group(1)}"
                return weight_name
            return weight_name

        layer_num = int(layer_match.group(1))

        # Determine component type
        wn = weight_name.lower()
        if any(x in wn for x in ["mlp", "fc", "gate", "up_proj", "down_proj",
                                   "dense_h_to_4h", "dense_4h_to_h", "c_fc"]):
            return f"L{layer_num}_MLP"
        elif any(x in wn for x in ["attn", "attention", "self_attn", "q_proj",
                                     "k_proj", "v_proj", "o_proj", "c_attn",
                                     "c_proj", "query_key_value"]):
            return f"L{layer_num}H"
        else:
            return f"L{layer_num}"

    def build_candidate_components(self) -> Set[str]:
        """
        Get all components containing DCAF-identified weights.

        Returns:
            Set of component names
        """
        components = set()
        for weight in self.weight_candidates:
            component = self.map_weight_to_component(weight)
            components.add(component)
        return components

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Causal Connectivity via Ablation
    # ─────────────────────────────────────────────────────────────

    def add_ablation_edges(
        self,
        ablated_component: str,
        delta: ActivationDelta,
    ) -> None:
        """
        Add edges based on ablation effects.

        After ablating a component, record which OTHER components changed.
        If ablating L10_MLP causes L12H7's activations to shift → edge L10_MLP → L12H7

        Args:
            ablated_component: Component that was ablated
            delta: Activation delta from the ablation
        """
        all_changes = delta.get_all_changes()

        for component, change in all_changes.items():
            if component != ablated_component and change > self.edge_threshold:
                self.graph.add_edge(
                    ablated_component,
                    component,
                    change,
                    "ablation",
                )

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Attention Flow Refinement
    # ─────────────────────────────────────────────────────────────

    def add_attention_edges(
        self,
        attention_patterns: Dict[str, torch.Tensor],
    ) -> None:
        """
        Add edges based on attention flow patterns.

        Attention heads write to the residual stream at their OWN position after
        attending to other positions. The flow is:
        1. L10H3 attends to tokens, writes signal at position P
        2. L12H7 later attends to position P to read that signal

        Track: which later heads attend to positions where earlier heads wrote strongly.

        Args:
            attention_patterns: Dict mapping head name to attention weights
        """
        # Group heads by layer
        heads_by_layer: Dict[int, List[str]] = {}
        for head_name in attention_patterns:
            match = re.match(r"L(\d+)H(\d+)", head_name)
            if match:
                layer = int(match.group(1))
                if layer not in heads_by_layer:
                    heads_by_layer[layer] = []
                heads_by_layer[layer].append(head_name)

        layers = sorted(heads_by_layer.keys())

        # For each pair of layers (earlier, later), check attention flow
        for i, earlier_layer in enumerate(layers):
            for later_layer in layers[i + 1 :]:
                earlier_heads = heads_by_layer[earlier_layer]
                later_heads = heads_by_layer[later_layer]

                for earlier_head in earlier_heads:
                    earlier_attn = attention_patterns.get(earlier_head)
                    if earlier_attn is None:
                        continue

                    # Get positions where earlier head wrote its output.
                    # Attention [seq_q, seq_k]: row i = query i's attention distribution.
                    # The head writes its output at QUERY positions (row sums = total
                    # attention mass originating from each position).
                    earlier_influence = earlier_attn.float().mean(dim=0)
                    write_positions = earlier_influence.sum(dim=1)  # [seq] — row sums

                    for later_head in later_heads:
                        later_attn = attention_patterns.get(later_head)
                        if later_attn is None:
                            continue

                        # Get positions that later head reads FROM (column sums =
                        # which key positions receive the most attention).
                        later_influence = later_attn.float().mean(dim=0)
                        read_positions = later_influence.sum(dim=0)  # [seq] — col sums

                        # Compute overlap: does later head read from where earlier wrote?
                        # Normalize both to probability distributions
                        if write_positions.sum() > 0 and read_positions.sum() > 0:
                            write_probs = write_positions / write_positions.sum()
                            read_probs = read_positions / read_positions.sum()

                            # Compute overlap (cosine-like similarity)
                            overlap = (write_probs * read_probs).sum().item()

                            if overlap > self.edge_threshold:
                                # Add edge with reduced weight (correlational, not causal)
                                edge_weight = overlap * self.attention_weight
                                self.graph.add_edge(
                                    earlier_head,
                                    later_head,
                                    edge_weight,
                                    "attention",
                                )

    # ─────────────────────────────────────────────────────────────
    # STEP 3.5: Generation Steering Edges (Delta-of-Deltas)
    # ─────────────────────────────────────────────────────────────

    def set_generation_properties(
        self,
        generation_threshold: float = TAU_EDGE,
    ) -> None:
        """
        Set node properties for generation steering circuits using delta-of-deltas.

        Identifies components that differentiate safe vs unsafe generation paths:
        - pre_delta = |pre_safe - pre_unsafe|  (small, no differentiation)
        - post_delta = |post_safe - post_unsafe|  (large, strong differentiation)
        - circuit_signal = post_delta - pre_delta

        Components with high circuit_signal are involved in generation steering.

        **Node Property Design:**
        Generation steering is represented as node properties (generation_score,
        is_generation_steering), not edges. This is semantically correct because
        generation steering is an intrinsic property of a component, not a
        directional connection between components.

        This differs from edges:
        - Ablation edges: component A → component B (causal influence)
        - Attention edges: head A → head B (information flow)
        - Generation properties: intrinsic to the node itself

        Args:
            generation_threshold: Minimum signal strength to mark as generation steering.
                Defaults to edge_threshold for consistency with edge detection.
        """
        pre_snapshot = self.training_delta.before
        post_snapshot = self.training_delta.after

        # Check if generation activations are available
        if not pre_snapshot.generation_activations or not post_snapshot.generation_activations:
            logger.debug("No generation activations available, skipping generation edges")
            return

        # Compute delta-of-deltas for each component
        component_signals: Dict[str, float] = {}

        for prompt in pre_snapshot.generation_activations:
            if prompt not in post_snapshot.generation_activations:
                continue

            pre_gen = pre_snapshot.generation_activations[prompt]
            post_gen = post_snapshot.generation_activations[prompt]

            # Process MLP components
            for component in set(pre_gen.safe_mlp.keys()) & set(post_gen.safe_mlp.keys()):
                # Get tensors
                pre_safe = pre_gen.safe_mlp[component]
                pre_unsafe = pre_gen.unsafe_mlp.get(component)
                post_safe = post_gen.safe_mlp[component]
                post_unsafe = post_gen.unsafe_mlp.get(component)

                if pre_unsafe is None or post_unsafe is None:
                    continue

                # Validate shapes match
                if pre_safe.shape != pre_unsafe.shape:
                    logger.warning(
                        f"Shape mismatch for {component} pre-training: "
                        f"safe {pre_safe.shape} vs unsafe {pre_unsafe.shape}, skipping"
                    )
                    continue
                if post_safe.shape != post_unsafe.shape:
                    logger.warning(
                        f"Shape mismatch for {component} post-training: "
                        f"safe {post_safe.shape} vs unsafe {post_unsafe.shape}, skipping"
                    )
                    continue

                # Compute deltas
                pre_delta = (pre_safe - pre_unsafe).abs().mean().item()
                post_delta = (post_safe - post_unsafe).abs().mean().item()
                circuit_signal = post_delta - pre_delta

                # Accumulate signals across prompts
                if component not in component_signals:
                    component_signals[component] = 0.0
                component_signals[component] += circuit_signal

            # Process attention components
            for component in set(pre_gen.safe_attention.keys()) & set(post_gen.safe_attention.keys()):
                pre_safe = pre_gen.safe_attention[component]
                pre_unsafe = pre_gen.unsafe_attention.get(component)
                post_safe = post_gen.safe_attention[component]
                post_unsafe = post_gen.unsafe_attention.get(component)

                if pre_unsafe is None or post_unsafe is None:
                    continue

                # Validate shapes match
                if pre_safe.shape != pre_unsafe.shape:
                    logger.warning(
                        f"Shape mismatch for {component} pre-training: "
                        f"safe {pre_safe.shape} vs unsafe {pre_unsafe.shape}, skipping"
                    )
                    continue
                if post_safe.shape != post_unsafe.shape:
                    logger.warning(
                        f"Shape mismatch for {component} post-training: "
                        f"safe {post_safe.shape} vs unsafe {post_unsafe.shape}, skipping"
                    )
                    continue

                pre_delta = (pre_safe - pre_unsafe).abs().mean().item()
                post_delta = (post_safe - post_unsafe).abs().mean().item()
                circuit_signal = post_delta - pre_delta

                if component not in component_signals:
                    component_signals[component] = 0.0
                component_signals[component] += circuit_signal

        # Normalize by number of prompts
        num_prompts = len(pre_snapshot.generation_activations)
        if num_prompts > 0:
            for component in component_signals:
                component_signals[component] /= num_prompts

        # Use generation_threshold parameter if provided, else fallback to edge_threshold
        threshold = generation_threshold if generation_threshold is not None else self.edge_threshold

        # Set node properties for components with strong generation steering
        num_steering = 0
        for component, signal in component_signals.items():
            # Ensure node exists
            if component not in self.graph.nodes:
                self.graph.add_node(component)

            # Set generation score for all components (even if below threshold)
            self.graph.nodes[component].generation_score = signal

            # Mark as generation steering if above threshold
            if signal > threshold:
                self.graph.nodes[component].is_generation_steering = True
                num_steering += 1

        if num_steering > 0:
            logger.info(f"Identified {num_steering} generation steering components")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Graph Construction
    # ─────────────────────────────────────────────────────────────

    def build_graph(self) -> None:
        """
        Build directed graph of component connectivity.

        Nodes = components containing DCAF-identified weights
        Edges = causal influence from ablation + correlational from attention + generation steering
        """
        # Add nodes from weight candidates
        for weight in self.weight_candidates:
            component = self.map_weight_to_component(weight)
            self.graph.add_node(component)

        # Add ablation edges (causal)
        for weight, delta in self.ablation_deltas.items():
            component = self.map_weight_to_component(weight)
            self.add_ablation_edges(component, delta)

        # Add attention edges (correlational, weighted lower)
        if self.training_delta.after.attention_patterns:
            self.add_attention_edges(self.training_delta.after.attention_patterns)

        # Set generation steering properties (delta-of-deltas)
        if self.training_delta.after.generation_activations:
            self.set_generation_properties()

        logger.info(
            f"Built graph with {len(self.graph.nodes)} nodes and "
            f"{len(self.graph.edges)} edges"
        )

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Circuit Extraction (3 clustering methods)
    # ─────────────────────────────────────────────────────────────

    def extract_circuits_disjoint(self) -> List[Circuit]:
        """
        Option 1: Disjoint subgraphs.

        Find weakly connected components in the graph.
        If subgraph A doesn't affect subgraph B's activations, they're separate circuits.

        Returns:
            List of circuits from connected components
        """
        connected = self.graph.get_connected_components()
        circuits = []

        for i, nodes in enumerate(connected):
            circuit = self._subgraph_to_circuit(nodes, "disjoint", i)
            circuits.append(circuit)

        return circuits

    def extract_circuits_probe_response(
        self,
        probe_activations: Dict[str, Dict[str, float]],
        n_clusters: Optional[int] = None,
        linkage_threshold: float = 0.5,
    ) -> List[Circuit]:
        """
        Option 2: Cluster by probe co-activation patterns.

        Components that activate together for same probes → same circuit.
        Uses hierarchical (agglomerative) clustering on co-activation matrix.

        Args:
            probe_activations: Dict {probe: {component: activation}}
            n_clusters: Number of clusters (if None, use threshold)
            linkage_threshold: Distance threshold for cutting dendrogram

        Returns:
            List of circuits from clustering
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform

        # Get all components in the graph
        components = sorted(self.graph.nodes)
        if len(components) < 2:
            return [self._subgraph_to_circuit(set(components), "probe-response", 0)]

        probes = list(probe_activations.keys())
        if not probes:
            logger.warning("No probe activations provided, falling back to disjoint")
            return self.extract_circuits_disjoint()

        # Build activation matrix [n_components, n_probes]
        activation_matrix = np.zeros((len(components), len(probes)))
        for i, component in enumerate(components):
            for j, probe in enumerate(probes):
                activation_matrix[i, j] = probe_activations.get(probe, {}).get(
                    component, 0.0
                )

        # Compute correlation distance matrix
        if activation_matrix.shape[1] > 1:
            # Pearson correlation matrix
            corr_matrix = np.corrcoef(activation_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            # Convert correlation to distance (1 - correlation)
            dist_matrix = 1 - corr_matrix
            np.fill_diagonal(dist_matrix, 0)
        else:
            # Not enough probes for correlation, use euclidean
            dist_matrix = squareform(pdist(activation_matrix))

        # Hierarchical clustering with Ward linkage
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method="ward")

        # Cut dendrogram to get clusters
        if n_clusters is not None:
            labels = fcluster(Z, n_clusters, criterion="maxclust")
        else:
            labels = fcluster(Z, linkage_threshold, criterion="distance")

        # Group components by cluster
        clusters: Dict[int, Set[str]] = {}
        for component, label in zip(components, labels):
            if label not in clusters:
                clusters[label] = set()
            clusters[label].add(component)

        # Convert clusters to circuits
        circuits = []
        for i, (label, nodes) in enumerate(sorted(clusters.items())):
            circuit = self._subgraph_to_circuit(nodes, "probe-response", i)
            circuits.append(circuit)

        return circuits

    def extract_circuits_functional(
        self,
        harmful_activations: Dict[str, float],
        neutral_activations: Dict[str, float],
        activation_threshold: float = TAU_EDGE,
    ) -> List[Circuit]:
        """
        Option 3: Cluster by harmful vs neutral response.

        Classification logic for each component:
        - SAFETY: harmful_activation > threshold AND neutral_activation < threshold
        - CAPABILITY: neutral_activation > threshold AND harmful_activation < threshold
        - SHARED: harmful_activation > threshold AND neutral_activation > threshold
        - INACTIVE: both < threshold (exclude from circuits)

        Args:
            harmful_activations: Component activations on harmful probes
            neutral_activations: Component activations on neutral probes
            activation_threshold: Min activation to count as "active"

        Returns:
            List of circuits (only SAFETY components, SHARED flagged)
        """
        classifications: Dict[str, _FunctionalClusterLabel] = {}

        for component in self.graph.nodes:
            harmful_act = harmful_activations.get(component, 0.0)
            neutral_act = neutral_activations.get(component, 0.0)

            if harmful_act > activation_threshold and neutral_act < activation_threshold:
                category = "safety"
            elif neutral_act > activation_threshold and harmful_act < activation_threshold:
                category = "capability"
            elif harmful_act > activation_threshold and neutral_act > activation_threshold:
                category = "shared"
            else:
                category = "inactive"

            classifications[component] = _FunctionalClusterLabel(
                component=component,
                harmful_activation=harmful_act,
                neutral_activation=neutral_act,
                category=category,
            )

        # Build circuits from SAFETY and SHARED components
        safety_components = {
            c.component
            for c in classifications.values()
            if c.category in ("safety", "shared")
        }

        if not safety_components:
            logger.warning("No safety or shared components found")
            return []

        # Run disjoint clustering on safety components only
        subgraph = self.graph.get_subgraph(safety_components)
        connected = subgraph.get_connected_components()

        circuits = []
        for i, nodes in enumerate(connected):
            circuit = self._subgraph_to_circuit(nodes, "functional", i)
            circuits.append(circuit)

        return circuits

    def _subgraph_to_circuit(
        self,
        nodes: Set[str],
        method: str,
        index: int,
    ) -> Circuit:
        """Convert a set of nodes to a Circuit object."""
        # Get edges within this subgraph
        edges = self.graph.get_edges_in_subgraph(nodes)

        # Get nodes with properties (generation_score, is_generation_steering)
        circuit_nodes = {}
        for node_name in nodes:
            if node_name in self.graph.nodes:
                circuit_nodes[node_name] = self.graph.nodes[node_name]

        # Map components back to weights
        weight_params = []
        for weight in self.weight_candidates:
            component = self.map_weight_to_component(weight)
            if component in nodes:
                weight_params.append(weight)

        # Compute flow (topological sort)
        flow = self.graph.topological_sort(nodes)

        return Circuit(
            name=f"circuit_{index}",
            components=list(nodes),
            weight_params=weight_params,
            flow=flow,
            edges=edges,
            nodes=circuit_nodes,
            clustering_method=method,
            validation=None,
            confidence=0.0,
        )

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Circuit Validation
    # ─────────────────────────────────────────────────────────────

    def get_ablation_impact(self, weight: str) -> float:
        """
        Get the safety impact of ablating a single weight.

        Args:
            weight: Weight parameter name

        Returns:
            Impact score (higher = more impact on safety)
        """
        if weight in self._ablation_impacts:
            return self._ablation_impacts[weight]

        # Compute from ablation delta if available
        if weight in self.ablation_deltas:
            delta = self.ablation_deltas[weight]
            all_changes = delta.get_all_changes()
            impact = sum(all_changes.values())
            self._ablation_impacts[weight] = impact
            return impact

        return 0.0

    def validate_circuit(
        self,
        circuit: Circuit,
        state_manager=None,
        measure_safety_fn=None,
    ) -> Optional[CircuitValidation]:
        """
        Validate that circuit exhibits superadditive effects.

        Ablate ALL weights in circuit simultaneously and check if safety breaks
        MORE than sum of individual ablations.

        Args:
            circuit: Circuit to validate
            state_manager: Optional ModelStateManager for actual ablation
            measure_safety_fn: Optional function to measure safety impact

        Returns:
            CircuitValidation with superadditivity test results, or None if
            state_manager/measure_safety_fn not provided (actual measurement required)
        """
        # If we can't actually measure, don't fake it with heuristics
        if state_manager is None or measure_safety_fn is None:
            logger.debug(
                f"Skipping validation for circuit '{circuit.name}' - "
                "state_manager and measure_safety_fn required for actual measurement"
            )
            return None

        # Sum of individual impacts
        individual_sum = sum(
            self.get_ablation_impact(w) for w in circuit.weight_params
        )

        # Ablate entire circuit and measure actual impact
        state_manager.ablate_params(circuit.weight_params)
        whole_impact = measure_safety_fn()
        state_manager.restore_params(circuit.weight_params)

        # Check superadditivity (10% threshold)
        superadditive = whole_impact > individual_sum * 1.1

        return CircuitValidation(
            individual_ablation_impact=individual_sum,
            whole_circuit_ablation_impact=whole_impact,
            superadditive=superadditive,
        )

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Flow Computation
    # ─────────────────────────────────────────────────────────────

    def compute_flow(self, circuit: Circuit) -> List[str]:
        """
        Compute information flow order for circuit.

        Topological sort of circuit subgraph respecting:
        - Layer order as tiebreaker (L10 before L11 before L12)
        - Within layer: attention before MLP (transformer processing order)

        Args:
            circuit: Circuit to compute flow for

        Returns:
            Components sorted in flow order
        """
        return self.graph.topological_sort(set(circuit.components))

    def add_weight_classifications(
        self,
        circuits: List[Circuit],
        weight_classifications: Dict[str, "WeightClassification"],
    ) -> None:
        """
        Add weight classifications to circuits and compute circuit types.

        Args:
            circuits: List of circuits to populate with classifications
            weight_classifications: Dict mapping weight names to classifications
        """
        for circuit in circuits:
            # Add classifications for weights in this circuit
            for weight in circuit.weight_params:
                if weight in weight_classifications:
                    circuit.weight_classifications[weight] = weight_classifications[weight]

            # Compute circuit type from majority
            if circuit.weight_classifications:
                circuit.circuit_type = circuit.compute_circuit_type()

    # ─────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────

    def identify_circuits(
        self,
        method: ClusteringMethod = "disjoint",
        state_manager=None,
        measure_safety_fn=None,
        probe_activations: Optional[Dict[str, Dict[str, float]]] = None,
        harmful_activations: Optional[Dict[str, float]] = None,
        neutral_activations: Optional[Dict[str, float]] = None,
    ) -> CircuitAnalysisResults:
        """
        Full circuit identification pipeline.

        Args:
            method: Clustering method ("disjoint", "probe-response", "functional")
            state_manager: Optional ModelStateManager for validation
            measure_safety_fn: Optional function to measure safety impact
            probe_activations: Required for "probe-response" method
            harmful_activations: Required for "functional" method
            neutral_activations: Required for "functional" method

        Returns:
            CircuitAnalysisResults with identified circuits
        """
        # Build the graph
        self.build_graph()

        # Extract circuits based on method
        if method == "disjoint":
            circuits = self.extract_circuits_disjoint()
        elif method == "probe-response":
            if probe_activations is None:
                raise ValueError("probe_activations required for probe-response method")
            circuits = self.extract_circuits_probe_response(probe_activations)
        elif method == "functional":
            if harmful_activations is None or neutral_activations is None:
                raise ValueError(
                    "harmful_activations and neutral_activations required for functional method"
                )
            circuits = self.extract_circuits_functional(
                harmful_activations, neutral_activations
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Validate and compute flow for each circuit
        for circuit in circuits:
            circuit.flow = self.compute_flow(circuit)
            circuit.validation = self.validate_circuit(
                circuit, state_manager, measure_safety_fn
            )
            # Compute confidence based on validation and size
            if circuit.validation and circuit.validation.superadditive:
                circuit.confidence = min(
                    0.9 + 0.1 * len(circuit.components) / 10, 1.0
                )
            else:
                circuit.confidence = 0.5 * len(circuit.components) / 10

        logger.info(f"Identified {len(circuits)} circuits using {method} method")

        return CircuitAnalysisResults(
            circuits=circuits,
            clustering_method=method,
            edge_threshold=self.edge_threshold,
            attention_weight=self.attention_weight,
            total_weight_candidates=len(self.weight_candidates),
            probe_set_name=self.probe_set.name,
        )


__all__ = ["CircuitIdentifier"]
