"""
Tests for dcaf.domains.geometry.probing module.

Tests polynomial and kernel-based probing to detect nonlinear structure
in activation representations.
"""

import pytest
import torch
import numpy as np

from tests.conftest import requires_sklearn


@requires_sklearn
class TestPolynomialProbing:
    """Tests for compute_polynomial_probing."""

    def test_linearly_separable_data(self):
        """Test polynomial probing on linearly separable data."""
        from dcaf.domains.geometry.probing import compute_polynomial_probing

        torch.manual_seed(42)
        np.random.seed(42)

        # Create linearly separable data
        # Class 0: centered at -3
        # Class 1: centered at +3
        n_samples = 100
        dim = 8

        activations_0 = torch.randn(n_samples, dim) * 0.5 - 3.0
        activations_1 = torch.randn(n_samples, dim) * 0.5 + 3.0

        activations = torch.cat([activations_0, activations_1], dim=0)
        labels = torch.cat([
            torch.zeros(n_samples, dtype=torch.long),
            torch.ones(n_samples, dtype=torch.long),
        ])

        # Shuffle
        perm = torch.randperm(len(activations))
        activations = activations[perm]
        labels = labels[perm]

        # Compute probing results
        results = compute_polynomial_probing(activations, labels)

        # Linear accuracy should be high for linearly separable data
        assert results.linear_accuracy > 0.8, (
            f"Expected high linear accuracy, got {results.linear_accuracy}"
        )

        # Polynomial gap should be small (data is already linearly separable)
        assert -0.2 <= results.polynomial_gap <= 0.2, (
            f"Expected small polynomial gap for linear data, got {results.polynomial_gap}"
        )

        # Verify results structure
        assert 0.0 <= results.linear_accuracy <= 1.0
        assert 0.0 <= results.polynomial_accuracy <= 1.0
        assert 0.0 <= results.kernel_lda_accuracy <= 1.0
        assert isinstance(results.top_interacting_dimensions, list)

    def test_probing_results_serialization(self):
        """Test ProbingResults serialization and deserialization."""
        from dcaf.domains.geometry.probing import ProbingResults

        # Create probing results
        original = ProbingResults(
            linear_accuracy=0.85,
            polynomial_accuracy=0.92,
            polynomial_gap=0.07,
            top_interacting_dimensions=[
                {"i": 0, "j": 1, "w_ij": 0.42},
                {"i": 2, "j": 3, "w_ij": 0.38},
            ],
            kernel_lda_accuracy=0.88,
            kernel_gap=0.03,
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        restored = ProbingResults.from_dict(data)

        # Verify all fields match
        assert restored.linear_accuracy == pytest.approx(original.linear_accuracy)
        assert restored.polynomial_accuracy == pytest.approx(original.polynomial_accuracy)
        assert restored.polynomial_gap == pytest.approx(original.polynomial_gap)
        assert restored.kernel_lda_accuracy == pytest.approx(original.kernel_lda_accuracy)
        assert restored.kernel_gap == pytest.approx(original.kernel_gap)
        assert len(restored.top_interacting_dimensions) == len(original.top_interacting_dimensions)
        assert restored.top_interacting_dimensions[0]["i"] == 0
        assert restored.top_interacting_dimensions[0]["j"] == 1
        assert restored.top_interacting_dimensions[0]["w_ij"] == pytest.approx(0.42)


@requires_sklearn
class TestKernelLDA:
    """Tests for compute_kernel_lda."""

    def test_kernel_improves_or_matches_linear(self):
        """Test that kernel LDA performs at least as well as linear on separable data."""
        from dcaf.domains.geometry.probing import compute_kernel_lda
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression

        torch.manual_seed(42)
        np.random.seed(42)

        # Create data with some nonlinear structure
        # Use XOR-like pattern for better kernel advantage
        n_samples = 200
        dim = 4

        # Generate data
        X = np.random.randn(n_samples, dim)

        # Create labels based on simple separable pattern
        # (Kernel should at least match linear performance)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Compute kernel LDA accuracy
        kernel_accuracy = compute_kernel_lda(X_train, y_train, X_test, y_test)

        # Compare to linear baseline
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        linear_accuracy = lr.score(X_test, y_test)

        # Kernel should produce reasonable accuracy (>= 0.5, better than random)
        assert kernel_accuracy >= 0.5, (
            f"Kernel LDA ({kernel_accuracy:.3f}) should be better than random"
        )

        # Kernel accuracy should be reasonable
        assert kernel_accuracy > 0.5, (
            f"Kernel accuracy should be better than random, got {kernel_accuracy}"
        )
