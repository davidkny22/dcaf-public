"""
Tests for CircuitGraph: directed graph representation for component connectivity.

Tests cover:
- Node and edge operations
- Connected components
- Topological sorting
- Subgraph extraction
- Serialization
"""

import pytest
import torch
from dcaf.circuit.graph import (
    CircuitGraph,
    CircuitNode,
    CircuitEdge,
)


# Set random seed for reproducibility
torch.manual_seed(42)


class TestCircuitNode:
    """Tests for CircuitNode dataclass."""

    def test_node_creation(self):
        """Test creating a circuit node."""
        node = CircuitNode(
            name="L10H3",
            generation_score=0.85,
            is_generation_steering=True,
        )

        assert node.name == "L10H3"
        assert node.generation_score == 0.85
        assert node.is_generation_steering is True

    def test_node_serialization(self):
        """Test node to_dict and from_dict round-trip."""
        node = CircuitNode(
            name="L10_MLP",
            generation_score=0.72,
            is_generation_steering=False,
        )

        data = node.to_dict()
        node2 = CircuitNode.from_dict(data)

        assert node2.name == node.name
        assert node2.generation_score == node.generation_score
        assert node2.is_generation_steering == node.is_generation_steering


class TestCircuitEdge:
    """Tests for CircuitEdge dataclass."""

    def test_edge_creation(self):
        """Test creating a circuit edge."""
        edge = CircuitEdge(
            source="L10H3",
            target="L11_MLP",
            weight=0.65,
            edge_type="ablation",
        )

        assert edge.source == "L10H3"
        assert edge.target == "L11_MLP"
        assert edge.weight == 0.65
        assert edge.edge_type == "ablation"

    def test_edge_serialization(self):
        """Test edge to_dict."""
        edge = CircuitEdge(
            source="L10H3",
            target="L11H5",
            weight=0.45,
            edge_type="attention",
        )

        data = edge.to_dict()

        assert data["source"] == "L10H3"
        assert data["target"] == "L11H5"
        assert data["weight"] == 0.45
        assert data["edge_type"] == "attention"


class TestCircuitGraphBasics:
    """Tests for basic graph operations."""

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = CircuitGraph()

        graph.add_node("L10H3", generation_score=0.85)

        assert "L10H3" in graph.nodes
        assert graph.nodes["L10H3"].generation_score == 0.85
        assert len(graph) == 1

    def test_add_node_updates_properties(self):
        """Test adding existing node updates properties."""
        graph = CircuitGraph()

        graph.add_node("L10H3", generation_score=0.5)
        graph.add_node("L10H3", generation_score=0.85, is_generation_steering=True)

        node = graph.nodes["L10H3"]
        assert node.generation_score == 0.85
        assert node.is_generation_steering is True

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.65, edge_type="ablation")

        assert ("L10H3", "L11_MLP") in graph.edges
        assert graph.edges[("L10H3", "L11_MLP")].weight == 0.65
        assert graph.edges[("L10H3", "L11_MLP")].edge_type == "ablation"

        # Nodes should be auto-created
        assert "L10H3" in graph.nodes
        assert "L11_MLP" in graph.nodes

    def test_add_edge_updates_weight(self):
        """Test adding edge with higher weight updates it."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.5, edge_type="ablation")
        graph.add_edge("L10H3", "L11_MLP", weight=0.8, edge_type="ablation")

        edge = graph.edges[("L10H3", "L11_MLP")]
        assert edge.weight == 0.8  # Updated to higher weight

    def test_add_edge_ignores_lower_weight(self):
        """Test adding edge with lower weight doesn't update."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.8, edge_type="ablation")
        graph.add_edge("L10H3", "L11_MLP", weight=0.5, edge_type="ablation")

        edge = graph.edges[("L10H3", "L11_MLP")]
        assert edge.weight == 0.8  # Kept higher weight

    def test_self_edges_ignored(self):
        """Test self-edges are ignored (no longer used)."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L10H3", weight=0.9, edge_type="ablation")

        # Self-edge should not be added
        assert ("L10H3", "L10H3") not in graph.edges

    def test_get_all_edges(self):
        """Test getting all edges."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L10H5", "L11_MLP", weight=0.5, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H7", weight=0.7, edge_type="attention")

        edges = graph.get_all_edges()

        assert len(edges) == 3
        assert all(isinstance(e, CircuitEdge) for e in edges)

    def test_len(self):
        """Test graph length is number of nodes."""
        graph = CircuitGraph()

        graph.add_node("L10H3")
        graph.add_node("L10H5")
        graph.add_node("L11_MLP")

        assert len(graph) == 3


class TestCircuitGraphNeighbors:
    """Tests for neighbor and edge queries."""

    def test_get_edges_from(self):
        """Test getting outgoing edges from a node."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L10H3", "L11H5", weight=0.5, edge_type="ablation")
        graph.add_edge("L10H5", "L11_MLP", weight=0.4, edge_type="ablation")

        edges = graph.get_edges_from("L10H3")

        assert len(edges) == 2
        targets = {e.target for e in edges}
        assert targets == {"L11_MLP", "L11H5"}

    def test_get_edges_to(self):
        """Test getting incoming edges to a node."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L10H5", "L11_MLP", weight=0.5, edge_type="ablation")
        graph.add_edge("L10H7", "L12_MLP", weight=0.4, edge_type="ablation")

        edges = graph.get_edges_to("L11_MLP")

        assert len(edges) == 2
        sources = {e.source for e in edges}
        assert sources == {"L10H3", "L10H5"}

    def test_get_edges_from_empty(self):
        """Test getting edges from node with no outgoing edges."""
        graph = CircuitGraph()

        graph.add_node("L10H3")

        edges = graph.get_edges_from("L10H3")
        assert edges == []

    def test_get_edges_to_empty(self):
        """Test getting edges to node with no incoming edges."""
        graph = CircuitGraph()

        graph.add_node("L10H3")

        edges = graph.get_edges_to("L10H3")
        assert edges == []


class TestConnectedComponents:
    """Tests for connected component analysis."""

    def test_single_component(self):
        """Test graph with single connected component."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H5", weight=0.5, edge_type="ablation")

        components = graph.get_connected_components()

        assert len(components) == 1
        assert components[0] == {"L10H3", "L11_MLP", "L12H5"}

    def test_multiple_components(self):
        """Test graph with multiple disconnected components."""
        graph = CircuitGraph()

        # Component 1
        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")

        # Component 2
        graph.add_edge("L12H5", "L13_MLP", weight=0.5, edge_type="ablation")

        # Isolated node (Component 3)
        graph.add_node("L14H7")

        components = graph.get_connected_components()

        assert len(components) == 3

        # Check each component
        comp_sets = [set(c) for c in components]
        assert {"L10H3", "L11_MLP"} in comp_sets
        assert {"L12H5", "L13_MLP"} in comp_sets
        assert {"L14H7"} in comp_sets

    def test_bidirectional_connectivity(self):
        """Test weakly connected components (treats edges as undirected)."""
        graph = CircuitGraph()

        # Create a cycle: A -> B -> C -> A
        graph.add_edge("A", "B", weight=0.5, edge_type="ablation")
        graph.add_edge("B", "C", weight=0.5, edge_type="ablation")
        graph.add_edge("C", "A", weight=0.5, edge_type="ablation")

        components = graph.get_connected_components()

        assert len(components) == 1
        assert components[0] == {"A", "B", "C"}

    def test_empty_graph(self):
        """Test connected components on empty graph."""
        graph = CircuitGraph()

        components = graph.get_connected_components()
        assert components == []


class TestTopologicalSort:
    """Tests for topological sorting."""

    def test_topological_sort_simple(self):
        """Test topological sort on simple DAG."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H5", weight=0.5, edge_type="ablation")

        sorted_nodes = graph.topological_sort()

        # Should respect dependency order
        assert sorted_nodes.index("L10H3") < sorted_nodes.index("L11_MLP")
        assert sorted_nodes.index("L11_MLP") < sorted_nodes.index("L12H5")

    def test_topological_sort_layer_order(self):
        """Test topological sort respects layer numbers."""
        graph = CircuitGraph()

        # Add nodes in random order
        graph.add_node("L12H5")
        graph.add_node("L10H3")
        graph.add_node("L11_MLP")

        sorted_nodes = graph.topological_sort()

        # Should be sorted by layer number
        assert sorted_nodes.index("L10H3") < sorted_nodes.index("L11_MLP")
        assert sorted_nodes.index("L11_MLP") < sorted_nodes.index("L12H5")

    def test_topological_sort_attention_before_mlp(self):
        """Test attention heads sorted before MLP in same layer."""
        graph = CircuitGraph()

        graph.add_node("L10_MLP")
        graph.add_node("L10H3")
        graph.add_node("L10H5")

        sorted_nodes = graph.topological_sort()

        # Attention heads should come before MLP in same layer
        mlp_index = sorted_nodes.index("L10_MLP")
        h3_index = sorted_nodes.index("L10H3")
        h5_index = sorted_nodes.index("L10H5")

        assert h3_index < mlp_index
        assert h5_index < mlp_index

    def test_topological_sort_subset(self):
        """Test topological sort on subset of nodes."""
        graph = CircuitGraph()

        graph.add_node("L10H3")
        graph.add_node("L11_MLP")
        graph.add_node("L12H5")
        graph.add_node("L13_MLP")

        # Sort only subset
        subset = {"L11_MLP", "L12H5"}
        sorted_nodes = graph.topological_sort(nodes=subset)

        assert len(sorted_nodes) == 2
        assert set(sorted_nodes) == subset
        assert sorted_nodes.index("L11_MLP") < sorted_nodes.index("L12H5")

    def test_topological_sort_with_cycle(self):
        """Test topological sort handles cycles gracefully."""
        graph = CircuitGraph()

        # Create cycle
        graph.add_edge("A", "B", weight=0.5, edge_type="ablation")
        graph.add_edge("B", "C", weight=0.5, edge_type="ablation")
        graph.add_edge("C", "A", weight=0.5, edge_type="ablation")

        # Should fall back to layer order (all nodes included)
        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) == 3
        assert set(sorted_nodes) == {"A", "B", "C"}


class TestSubgraph:
    """Tests for subgraph extraction."""

    def test_get_subgraph(self):
        """Test extracting a subgraph."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H5", weight=0.5, edge_type="ablation")
        graph.add_edge("L12H5", "L13_MLP", weight=0.4, edge_type="ablation")

        # Extract middle portion
        subgraph = graph.get_subgraph({"L11_MLP", "L12H5"})

        assert len(subgraph) == 2
        assert "L11_MLP" in subgraph.nodes
        assert "L12H5" in subgraph.nodes
        assert "L10H3" not in subgraph.nodes
        assert "L13_MLP" not in subgraph.nodes

        # Only edge between the two nodes should be preserved
        assert ("L11_MLP", "L12H5") in subgraph.edges
        assert len(subgraph.get_all_edges()) == 1

    def test_get_subgraph_preserves_properties(self):
        """Test subgraph preserves node properties."""
        graph = CircuitGraph()

        graph.add_node("L10H3", generation_score=0.85, is_generation_steering=True)
        graph.add_node("L11_MLP", generation_score=0.72)

        subgraph = graph.get_subgraph({"L10H3"})

        node = subgraph.nodes["L10H3"]
        assert node.generation_score == 0.85
        assert node.is_generation_steering is True

    def test_get_edges_in_subgraph(self):
        """Test getting edges within a node set."""
        graph = CircuitGraph()

        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H5", weight=0.5, edge_type="ablation")
        graph.add_edge("L12H5", "L13_MLP", weight=0.4, edge_type="ablation")

        nodes = {"L10H3", "L11_MLP", "L12H5"}
        edges = graph.get_edges_in_subgraph(nodes)

        assert len(edges) == 2
        sources = {e.source for e in edges}
        targets = {e.target for e in edges}

        assert sources.issubset(nodes)
        assert targets.issubset(nodes)


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict(self):
        """Test graph serialization to dict."""
        graph = CircuitGraph()

        graph.add_node("L10H3", generation_score=0.85, is_generation_steering=True)
        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")

        data = graph.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2  # L10H3 and L11_MLP
        assert len(data["edges"]) == 1

    def test_from_dict(self):
        """Test graph deserialization from dict."""
        data = {
            "nodes": [
                {
                    "name": "L10H3",
                    "generation_score": 0.85,
                    "is_generation_steering": True,
                },
                {
                    "name": "L11_MLP",
                    "generation_score": 0.0,
                    "is_generation_steering": False,
                },
            ],
            "edges": [
                {
                    "source": "L10H3",
                    "target": "L11_MLP",
                    "weight": 0.6,
                    "edge_type": "ablation",
                },
            ],
        }

        graph = CircuitGraph.from_dict(data)

        assert len(graph) == 2
        assert "L10H3" in graph.nodes
        assert "L11_MLP" in graph.nodes
        assert graph.nodes["L10H3"].generation_score == 0.85
        assert graph.nodes["L10H3"].is_generation_steering is True

        assert ("L10H3", "L11_MLP") in graph.edges
        assert graph.edges[("L10H3", "L11_MLP")].weight == 0.6

    def test_round_trip_serialization(self):
        """Test complete round-trip serialization."""
        graph = CircuitGraph()

        graph.add_node("L10H3", generation_score=0.85, is_generation_steering=True)
        graph.add_node("L10H5", generation_score=0.72)
        graph.add_edge("L10H3", "L11_MLP", weight=0.6, edge_type="ablation")
        graph.add_edge("L10H5", "L11_MLP", weight=0.5, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H7", weight=0.7, edge_type="attention")

        # Round-trip
        data = graph.to_dict()
        graph2 = CircuitGraph.from_dict(data)

        assert len(graph2) == len(graph)
        assert set(graph2.nodes.keys()) == set(graph.nodes.keys())
        assert len(graph2.edges) == len(graph.edges)

        # Check node properties preserved
        assert graph2.nodes["L10H3"].generation_score == 0.85
        assert graph2.nodes["L10H3"].is_generation_steering is True

        # Check edge properties preserved
        edge = graph2.edges[("L10H3", "L11_MLP")]
        assert edge.weight == 0.6
        assert edge.edge_type == "ablation"

    def test_from_dict_legacy_format(self):
        """Test loading from legacy format (nodes as strings)."""
        data = {
            "nodes": ["L10H3", "L11_MLP"],  # Legacy: just strings
            "edges": [
                {
                    "source": "L10H3",
                    "target": "L11_MLP",
                    "weight": 0.6,
                    "edge_type": "ablation",
                },
            ],
        }

        graph = CircuitGraph.from_dict(data)

        assert len(graph) == 2
        assert "L10H3" in graph.nodes
        assert "L11_MLP" in graph.nodes
        # Legacy nodes should have default properties
        assert graph.nodes["L10H3"].generation_score == 0.0


class TestCircuitGraphIntegration:
    """Integration tests for complete graph operations."""

    def test_complex_circuit(self):
        """Test building and analyzing a complex circuit."""
        graph = CircuitGraph()

        # Build circuit with multiple paths
        graph.add_node("L10H3", generation_score=0.85, is_generation_steering=True)
        graph.add_node("L10H5", generation_score=0.72)
        graph.add_edge("L10H3", "L11_MLP", weight=0.65, edge_type="ablation")
        graph.add_edge("L10H5", "L11_MLP", weight=0.55, edge_type="ablation")
        graph.add_edge("L11_MLP", "L12H7", weight=0.70, edge_type="ablation")
        graph.add_edge("L10H3", "L12H7", weight=0.45, edge_type="attention")

        # Verify structure
        assert len(graph) == 4

        # Check connectivity
        components = graph.get_connected_components()
        assert len(components) == 1

        # Check topological order
        sorted_nodes = graph.topological_sort()
        l10_indices = [i for i, n in enumerate(sorted_nodes) if n.startswith("L10")]
        l11_indices = [i for i, n in enumerate(sorted_nodes) if n.startswith("L11")]
        l12_indices = [i for i, n in enumerate(sorted_nodes) if n.startswith("L12")]

        assert max(l10_indices) < min(l11_indices)
        assert max(l11_indices) < min(l12_indices)

    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        graph = CircuitGraph()

        assert len(graph) == 0
        assert graph.get_all_edges() == []
        assert graph.get_connected_components() == []
        assert graph.topological_sort() == []
