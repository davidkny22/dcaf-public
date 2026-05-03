"""
Circuit graph data structures (sec:circuit-graph; def:behavioral-circuit-graph).

Provides the directed graph G = (V, E) where V = H_conf (confirmed components)
and edges represent causal relationships discovered through intervention.

Supports both causal edges (from ablation/steering) and correlational edges
(from attention pattern analysis).

    G = (V, {(k, k') : e(k, k') = 1})

See def:behavioral-circuit-graph for the complete specification.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CircuitNode:
    """
    A node in the circuit graph representing a model component.

    Nodes have intrinsic properties like generation steering score,
    rather than using self-edges to represent these properties.
    """

    name: str  # Component name (e.g., "L10H3", "L10_MLP")
    generation_score: float = 0.0  # Delta-of-deltas signal for generation steering
    is_generation_steering: bool = False  # Whether this component steers generation

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "generation_score": self.generation_score,
            "is_generation_steering": self.is_generation_steering,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitNode":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            generation_score=data.get("generation_score", 0.0),
            is_generation_steering=data.get("is_generation_steering", False),
        )


@dataclass
class CircuitEdge:
    """
    An edge in the circuit graph.

    Represents a connection between two components with a weight indicating
    strength of influence and edge type indicating how it was discovered.

    Note: Self-edges (source == target) should not be used. Generation steering
    is represented as node properties, not edges.
    """

    source: str  # Component name (e.g., "L10H3", "L10_MLP")
    target: str  # Component name (e.g., "L11_MLP", "L12H7")
    weight: float  # Strength of causal/correlational influence
    edge_type: str  # "ablation" (causal) or "attention" (correlational)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type,
        }


class CircuitGraph:
    """
    Directed graph of component connectivity.

    Nodes are CircuitNode objects with properties like generation_score.
    Edges represent actual connections between components (no self-edges).

    Supports:
    - Adding nodes (components) with properties
    - Adding edges (connections between different components)
    - Finding weakly connected components for disjoint clustering
    - Topological sorting respecting transformer layer order
    - Subgraph extraction for circuit isolation
    """

    def __init__(self):
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: Dict[Tuple[str, str], CircuitEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        component: str,
        generation_score: float = 0.0,
        is_generation_steering: bool = False,
    ) -> None:
        """
        Add a node (component) to the graph.

        Args:
            component: Component name
            generation_score: Delta-of-deltas signal for generation steering
            is_generation_steering: Whether this component steers generation
        """
        if component not in self.nodes:
            self.nodes[component] = CircuitNode(
                name=component,
                generation_score=generation_score,
                is_generation_steering=is_generation_steering,
            )
        else:
            # Update properties if already exists
            if generation_score != 0.0:
                self.nodes[component].generation_score = generation_score
            if is_generation_steering:
                self.nodes[component].is_generation_steering = is_generation_steering

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        edge_type: str,
    ) -> None:
        """
        Add a directed edge to the graph.

        If edge already exists, updates weight if new weight is higher.

        Args:
            source: Source component name
            target: Target component name
            weight: Edge weight (strength of influence)
            edge_type: "ablation" or "attention"

        Note:
            Self-edges (source == target) are no longer used. Generation steering
            should be set via node properties instead.
        """
        # Skip self-edges (generation steering is now a node property)
        if source == target:
            import logging
            logging.getLogger(__name__).debug(
                f"Ignoring self-edge for {source}. "
                "Use set_generation_properties() instead."
            )
            return

        self.add_node(source)
        self.add_node(target)

        key = (source, target)
        if key in self.edges:
            # Update if new weight is higher
            if weight > self.edges[key].weight:
                self.edges[key] = CircuitEdge(source, target, weight, edge_type)
        else:
            self.edges[key] = CircuitEdge(source, target, weight, edge_type)

        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)

    def get_edges_from(self, node: str) -> List[CircuitEdge]:
        """Get all edges originating from a node."""
        return [
            self.edges[(node, target)]
            for target in self._adjacency.get(node, set())
        ]

    def get_edges_to(self, node: str) -> List[CircuitEdge]:
        """Get all edges pointing to a node."""
        return [
            self.edges[(source, node)]
            for source in self._reverse_adjacency.get(node, set())
        ]

    def get_connected_components(self) -> List[Set[str]]:
        """
        Find weakly connected components in the graph.

        Treats graph as undirected for connectivity purposes.
        Used for disjoint clustering where separate subgraphs become
        separate circuits.

        Returns:
            List of sets, each containing node names in a connected component
        """
        visited = set()
        components = []

        def dfs(node: str, component: Set[str]) -> None:
            """Depth-first search treating graph as undirected."""
            if node in visited:
                return
            visited.add(node)
            component.add(node)

            # Follow edges in both directions
            for neighbor in self._adjacency.get(node, set()):
                dfs(neighbor, component)
            for neighbor in self._reverse_adjacency.get(node, set()):
                dfs(neighbor, component)

        # Iterate over node names (keys)
        for node_name in self.nodes.keys():
            if node_name not in visited:
                component = set()
                dfs(node_name, component)
                if component:
                    components.append(component)

        return components

    def topological_sort(self, nodes: Optional[Set[str]] = None) -> List[str]:
        """
        Topological sort of nodes respecting layer order.

        Uses transformer layer order as tiebreaker:
        - Lower layer numbers come first (L10 before L11)
        - Within layer: attention before MLP (standard transformer order)

        Args:
            nodes: Subset of nodes to sort (None for all nodes)

        Returns:
            Nodes sorted in topological/layer order
        """
        target_nodes = nodes if nodes is not None else set(self.nodes.keys())

        # Build in-degree map for target nodes
        in_degree = {node: 0 for node in target_nodes}
        for (src, tgt), edge in self.edges.items():
            if src in target_nodes and tgt in target_nodes:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # Sort by layer order as initial ordering
        def layer_order_key(node: str) -> Tuple[int, int]:
            """Extract (layer_num, component_type) for sorting."""
            layer_match = re.search(r"L(\d+)", node)
            layer_num = int(layer_match.group(1)) if layer_match else 999

            # Attention heads (H) before MLP
            if "H" in node and "_MLP" not in node:
                component_order = 0
            elif "_MLP" in node:
                component_order = 1
            else:
                component_order = 2

            return (layer_num, component_order)

        # Kahn's algorithm with layer-order tiebreaker
        result = []
        available = sorted(
            [n for n in target_nodes if in_degree.get(n, 0) == 0],
            key=layer_order_key,
        )

        while available:
            # Take the node with lowest layer order
            node = available.pop(0)
            result.append(node)

            # Update in-degrees
            for neighbor in self._adjacency.get(node, set()):
                if neighbor in target_nodes:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        # Insert maintaining layer order
                        available.append(neighbor)
                        available.sort(key=layer_order_key)

        # If cycle detected, fall back to layer order
        if len(result) < len(target_nodes):
            remaining = target_nodes - set(result)
            result.extend(sorted(remaining, key=layer_order_key))

        return result

    def get_subgraph(self, nodes: Set[str]) -> "CircuitGraph":
        """
        Extract a subgraph containing only the specified nodes.

        Preserves node properties (generation_score, is_generation_steering).

        Args:
            nodes: Set of nodes to include

        Returns:
            New CircuitGraph with only specified nodes and edges between them
        """
        subgraph = CircuitGraph()

        # Copy nodes with their properties
        for node_name in nodes:
            if node_name in self.nodes:
                node = self.nodes[node_name]
                subgraph.nodes[node_name] = CircuitNode(
                    name=node.name,
                    generation_score=node.generation_score,
                    is_generation_steering=node.is_generation_steering,
                )

        # Copy edges
        for (src, tgt), edge in self.edges.items():
            if src in nodes and tgt in nodes:
                subgraph.add_edge(src, tgt, edge.weight, edge.edge_type)

        return subgraph

    def get_all_edges(self) -> List[CircuitEdge]:
        """Get all edges in the graph."""
        return list(self.edges.values())

    def get_edges_in_subgraph(self, nodes: Set[str]) -> List[CircuitEdge]:
        """Get edges that connect nodes within the specified set."""
        return [
            edge
            for (src, tgt), edge in self.edges.items()
            if src in nodes and tgt in nodes
        ]

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitGraph":
        """Load from dictionary."""
        graph = cls()

        # Load nodes with properties
        for node_data in data.get("nodes", []):
            if isinstance(node_data, str):
                # Legacy format: just node name
                graph.add_node(node_data)
            else:
                # New format: full node data
                node = CircuitNode.from_dict(node_data)
                graph.nodes[node.name] = node

        # Load edges
        for edge_data in data.get("edges", []):
            graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                edge_data["weight"],
                edge_data["edge_type"],
            )
        return graph

__all__ = ["CircuitNode", "CircuitEdge", "CircuitGraph"]
