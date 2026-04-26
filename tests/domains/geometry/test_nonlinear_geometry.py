"""
Tests for dcaf.domains.geometry.nonlinear module.

Tests nonlinear representation diagnostics including PaCMAP embeddings,
silhouette scores, and Procrustes alignment.
"""

import pytest
import torch
import numpy as np

from tests.conftest import requires_pacmap, requires_scipy, MockSignal


@requires_pacmap
class TestPaCMAPActivations:
    """Tests for compute_pacmap_on_activations."""

    def test_separated_clusters_high_silhouette(self):
        """Test PaCMAP silhouette score on well-separated clusters."""
        from dcaf.domains.geometry.nonlinear import compute_pacmap_on_activations

        torch.manual_seed(42)
        np.random.seed(42)

        # Create well-separated clusters: T+ at +5, T- at -5
        # Use 20+ samples per cluster for PaCMAP stability
        n_samples = 25
        dim = 16

        # T+ cluster centered at +5
        A_plus = torch.randn(n_samples, dim) * 0.5 + 5.0
        # T- cluster centered at -5
        A_minus = torch.randn(n_samples, dim) * 0.5 - 5.0

        # Create signals
        signals = [
            MockSignal(id="t1_pref_target", cluster="+"),
            MockSignal(id="t6_pref_opposite", cluster="-"),
        ]

        # Create activation snapshots
        activation_snapshots = {
            "t1_pref_target": A_plus,
            "t6_pref_opposite": A_minus,
        }

        # Compute silhouette
        score = compute_pacmap_on_activations(
            component="L10H3",
            activation_snapshots=activation_snapshots,
            signals=signals,
        )

        # Well-separated clusters should have high silhouette (> 0.5)
        assert score > 0.5, f"Expected high silhouette for separated clusters, got {score}"

    def test_insufficient_samples_returns_zero(self):
        """Test that insufficient signals returns 0.0."""
        from dcaf.domains.geometry.nonlinear import compute_pacmap_on_activations

        torch.manual_seed(42)

        # Only one cluster provided
        signals = [
            MockSignal(id="t1_pref_target", cluster="+"),
        ]

        activation_snapshots = {
            "t1_pref_target": torch.randn(10, 16),
        }

        score = compute_pacmap_on_activations(
            component="L10H3",
            activation_snapshots=activation_snapshots,
            signals=signals,
        )

        assert score == 0.0, "Expected 0.0 for insufficient signals"


@requires_pacmap
class TestPaCMAPDeltas:
    """Tests for compute_pacmap_on_deltas."""

    def test_delta_silhouette(self):
        """Test PaCMAP silhouette on activation deltas."""
        from dcaf.domains.geometry.nonlinear import compute_pacmap_on_deltas

        torch.manual_seed(42)
        np.random.seed(42)

        # Create delta vectors: T+ positive changes, T- negative changes
        n_deltas = 20
        dim = 16

        # T+ deltas: positive changes
        delta_plus = torch.randn(n_deltas, dim) * 0.5 + 3.0
        # T- deltas: negative changes
        delta_minus = torch.randn(n_deltas, dim) * 0.5 - 3.0

        signals = [
            MockSignal(id="t1_pref_target", cluster="+"),
            MockSignal(id="t6_pref_opposite", cluster="-"),
        ]

        activation_deltas = {
            "t1_pref_target": delta_plus,
            "t6_pref_opposite": delta_minus,
        }

        score = compute_pacmap_on_deltas(
            component="L10H3",
            activation_deltas=activation_deltas,
            signals=signals,
        )

        # Well-separated deltas should have positive silhouette
        assert score > 0.0, f"Expected positive silhouette for separated deltas, got {score}"


@requires_scipy
@requires_pacmap
class TestProcrustes:
    """Tests for compute_procrustes_alignment."""

    def test_procrustes_alignment(self):
        """Test Procrustes structural mirror score with opposing embeddings."""
        from dcaf.domains.geometry.nonlinear import compute_procrustes_alignment

        torch.manual_seed(42)
        np.random.seed(42)

        # PaCMAP default n_neighbors=10, so each cluster needs at least 12 samples
        dim = 16
        n_per_cluster = 15

        signals = []
        activation_deltas = {}

        for i in range(n_per_cluster):
            pid = f"plus_{i}"
            mid = f"minus_{i}"
            signals.append(MockSignal(id=pid, cluster="+"))
            signals.append(MockSignal(id=mid, cluster="-"))
            activation_deltas[pid] = torch.randn(dim) * 0.5 + 2.0
            activation_deltas[mid] = torch.randn(dim) * 0.5 - 2.0

        score = compute_procrustes_alignment(
            component="L10H3",
            activation_deltas=activation_deltas,
            signals=signals,
        )

        # Score should be in [0, 1]
        assert 0.0 <= score <= 1.0, f"Procrustes score should be in [0, 1], got {score}"


class TestNonlinearDiagnostics:
    """Tests for NonlinearDiagnostics dataclass and compute_nonlinear_diagnostics."""

    def test_not_triggered_above_threshold(self):
        """Test that diagnostics are not triggered when LRS >= threshold."""
        from dcaf.domains.geometry.nonlinear import compute_nonlinear_diagnostics

        torch.manual_seed(42)

        # LRS above threshold should return None
        result = compute_nonlinear_diagnostics(
            component="L10H3",
            lrs=0.6,  # Above default threshold of 0.4
            activation_snapshots={},
            activation_deltas={},
            signals=[],
            lrs_threshold=0.4,
        )

        assert result is None, "Expected None when LRS >= threshold"

    def test_serialization_round_trip(self):
        """Test NonlinearDiagnostics serialization and deserialization."""
        from dcaf.domains.geometry.nonlinear import NonlinearDiagnostics

        # Create a diagnostic result
        original = NonlinearDiagnostics(
            triggered=True,
            lrs=0.35,
            pacmap_A_silhouette=0.72,
            pacmap_deltaA_silhouette=0.68,
            procrustes_structural_mirror_score=0.81,
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        restored = NonlinearDiagnostics.from_dict(data)

        # Verify all fields match
        assert restored.triggered == original.triggered
        assert restored.lrs == pytest.approx(original.lrs)
        assert restored.pacmap_A_silhouette == pytest.approx(original.pacmap_A_silhouette)
        assert restored.pacmap_deltaA_silhouette == pytest.approx(original.pacmap_deltaA_silhouette)
        assert restored.procrustes_structural_mirror_score == pytest.approx(
            original.procrustes_structural_mirror_score
        )
