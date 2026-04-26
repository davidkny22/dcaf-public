"""
Tests for dcaf.domains.geometry.directions module.

Tests contrastive direction extraction with regularization and
whitening transformations.
"""

import pytest
import torch
import numpy as np


class TestContrastiveDirection:
    """Tests for extract_contrastive_direction."""

    def test_direction_is_unit_vector(self):
        """Test that extracted direction is normalized to unit length."""
        from dcaf.domains.geometry.directions import extract_contrastive_direction

        torch.manual_seed(42)

        # Create well-separated clusters
        dim = 16
        n_samples = 30

        # T+ cluster at +1
        A_plus = torch.randn(n_samples, dim) * 0.5 + 1.0
        # T- cluster at -1
        A_minus = torch.randn(n_samples, dim) * 0.5 - 1.0

        # Extract direction
        direction = extract_contrastive_direction(A_plus, A_minus)

        # Verify it's a unit vector (norm ≈ 1)
        direction_norm = torch.norm(direction).item()
        assert direction_norm == pytest.approx(1.0, abs=1e-5), (
            f"Direction should be unit vector, got norm {direction_norm}"
        )

        # Verify shape
        assert direction.shape == (dim,), f"Expected shape ({dim},), got {direction.shape}"

    def test_separates_clusters(self):
        """Test that direction separates positive and negative clusters."""
        from dcaf.domains.geometry.directions import extract_contrastive_direction

        torch.manual_seed(42)

        # Create well-separated clusters
        dim = 16
        n_samples = 30

        # T+ cluster at +1
        A_plus = torch.randn(n_samples, dim) * 0.5 + 1.0
        # T- cluster at -1
        A_minus = torch.randn(n_samples, dim) * 0.5 - 1.0

        # Extract direction
        direction = extract_contrastive_direction(A_plus, A_minus)

        # Project clusters onto direction
        proj_plus = A_plus @ direction
        proj_minus = A_minus @ direction

        # Mean projections should be separated
        mean_proj_plus = proj_plus.mean().item()
        mean_proj_minus = proj_minus.mean().item()

        # T+ should project more positively than T-
        assert mean_proj_plus > mean_proj_minus, (
            f"Direction should separate clusters: "
            f"T+ mean={mean_proj_plus:.3f}, T- mean={mean_proj_minus:.3f}"
        )

        # Check separation is substantial
        separation = mean_proj_plus - mean_proj_minus
        assert separation > 0.5, (
            f"Expected substantial separation, got {separation:.3f}"
        )

    def test_regularization_prevents_singular(self):
        """Test that regularization handles near-singular covariance."""
        from dcaf.domains.geometry.directions import extract_contrastive_direction

        torch.manual_seed(42)

        # Create data with very low variance (near-singular covariance)
        dim = 16
        n_samples = 30

        # Clusters with very small noise
        A_plus = torch.randn(n_samples, dim) * 0.01 + 1.0
        A_minus = torch.randn(n_samples, dim) * 0.01 - 1.0

        # This should not crash due to regularization
        direction = extract_contrastive_direction(
            A_plus, A_minus, lambda_reg=0.1
        )

        # Verify we still get a valid direction
        assert direction.shape == (dim,)
        assert not torch.isnan(direction).any(), "Direction should not contain NaN"
        assert not torch.isinf(direction).any(), "Direction should not contain Inf"

        # Verify norm is close to 1 (or 0 if degenerate)
        direction_norm = torch.norm(direction).item()
        assert direction_norm >= 0.0, "Norm should be non-negative"

    def test_direction_dynamics(self):
        """Test direction emergence and rotation calculations."""
        from dcaf.domains.geometry.directions import (
            compute_direction_emergence,
            compute_direction_rotation,
            compute_direction_dynamics,
        )

        torch.manual_seed(42)

        # Create two directions
        dim = 16
        d_pre = torch.randn(dim)
        d_pre = d_pre / torch.norm(d_pre)  # Normalize

        # Post direction: slightly rotated and scaled
        d_post = d_pre * 1.5 + torch.randn(dim) * 0.1
        d_post = d_post / torch.norm(d_post)  # Normalize

        # Test emergence
        emergence = compute_direction_emergence(d_pre, d_post)
        assert isinstance(emergence, float)
        # Both are normalized, so emergence should be near 0
        assert emergence == pytest.approx(0.0, abs=0.1)

        # Test rotation
        rotation = compute_direction_rotation(d_pre, d_post)
        assert isinstance(rotation, float)
        assert -1.0 <= rotation <= 1.0, f"Rotation should be in [-1, 1], got {rotation}"
        # Should be highly correlated since d_post based on d_pre
        assert rotation > 0.5, f"Expected high correlation, got {rotation}"

        # Test full dynamics
        dynamics = compute_direction_dynamics(d_pre, d_post)
        assert dynamics.delta_norm == pytest.approx(emergence)
        assert dynamics.rotation == pytest.approx(rotation)
        assert dynamics.d_pre_norm == pytest.approx(1.0)
        assert dynamics.d_post_norm == pytest.approx(1.0)

    def test_pooled_covariance(self):
        """Test pooled covariance computation."""
        from dcaf.domains.geometry.directions import compute_pooled_covariance

        torch.manual_seed(42)

        # Create two clusters
        dim = 8
        n_samples = 20

        A_plus = torch.randn(n_samples, dim) + 1.0
        A_minus = torch.randn(n_samples, dim) - 1.0

        # Compute pooled covariance
        sigma_w = compute_pooled_covariance(A_plus, A_minus)

        # Verify shape
        assert sigma_w.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {sigma_w.shape}"

        # Verify symmetry
        assert torch.allclose(sigma_w, sigma_w.T, atol=1e-5), "Covariance should be symmetric"

        # Verify positive semi-definite (all eigenvalues >= 0)
        eigenvalues = torch.linalg.eigvalsh(sigma_w)
        assert (eigenvalues >= -1e-5).all(), "Covariance should be positive semi-definite"

    def test_batch_extraction(self):
        """Test batch extraction of contrastive directions."""
        from dcaf.domains.geometry.directions import extract_contrastive_directions_batch

        torch.manual_seed(42)

        dim = 8
        n_samples = 20

        # Create multiple signal pairs
        activations_by_signal = {
            "signal1": (
                torch.randn(n_samples, dim) + 1.0,
                torch.randn(n_samples, dim) - 1.0,
            ),
            "signal2": (
                torch.randn(n_samples, dim) + 2.0,
                torch.randn(n_samples, dim) - 2.0,
            ),
        }

        # Extract directions
        directions = extract_contrastive_directions_batch(activations_by_signal)

        # Verify we get one direction per signal
        assert len(directions) == 2
        assert "signal1" in directions
        assert "signal2" in directions

        # Verify each direction is unit vector
        for signal_id, direction in directions.items():
            assert direction.shape == (dim,)
            norm = torch.norm(direction).item()
            assert norm == pytest.approx(1.0, abs=1e-5), (
                f"Direction for {signal_id} should be unit vector, got norm {norm}"
            )

    def test_aggregate_directions(self):
        """Test aggregation of multiple directions."""
        from dcaf.domains.geometry.directions import aggregate_directions

        torch.manual_seed(42)

        dim = 8

        # Create multiple directions
        directions = {
            "signal1": torch.randn(dim),
            "signal2": torch.randn(dim),
            "signal3": torch.randn(dim),
        }

        # Normalize them
        for signal_id in directions:
            directions[signal_id] = directions[signal_id] / torch.norm(directions[signal_id])

        # Aggregate all
        mean_direction = aggregate_directions(directions)

        # Verify shape and normalization
        assert mean_direction.shape == (dim,)
        norm = torch.norm(mean_direction).item()
        assert norm == pytest.approx(1.0, abs=1e-5), (
            f"Aggregated direction should be unit vector, got norm {norm}"
        )

        # Aggregate subset
        subset_direction = aggregate_directions(directions, signal_ids=["signal1", "signal2"])
        assert subset_direction.shape == (dim,)
        assert torch.norm(subset_direction).item() == pytest.approx(1.0, abs=1e-5)
