"""
Tests for Linear Representation Score (LRS) computation.

Tests cover power_mean and compute_lrs following the specification:
LRS(k) = [(Σ_{i=1}^6 w_i · (x_i + ε)^p) / (Σ_{i=1}^6 w_i)]^{1/p}
"""

import pytest
import torch
from dcaf.domains.geometry.lrs import (
    power_mean,
    compute_lrs,
    compute_lrs_from_breakdown,
    compute_lrs_batch,
    is_strong_representation,
    is_weak_representation,
    LRSBreakdown,
    LRSResult,
)


# Set random seed for reproducibility
torch.manual_seed(42)


class TestPowerMean:
    """Tests for power_mean function."""

    def test_arithmetic_mean(self):
        """Test p=1 gives arithmetic mean."""
        values = [0.6, 0.8, 0.7]
        weights = [1.0, 1.0, 1.0]
        result = power_mean(values, weights, p=1.0, epsilon=0.0)
        expected = (0.6 + 0.8 + 0.7) / 3.0
        assert result == pytest.approx(expected, abs=1e-5)

    def test_harmonic_mean(self):
        """Test p=-1 gives harmonic mean (with epsilon=0)."""
        values = [0.6, 0.8, 0.7]
        weights = [1.0, 1.0, 1.0]
        result = power_mean(values, weights, p=-1.0, epsilon=0.0)
        # Harmonic mean: 3 / (1/0.6 + 1/0.8 + 1/0.7)
        expected = 3.0 / (1.0/0.6 + 1.0/0.8 + 1.0/0.7)
        assert result == pytest.approx(expected, abs=1e-5)

    def test_p_half_default(self):
        """Test p=0.5 (default LRS power)."""
        values = [0.64, 0.81]  # Perfect squares for clarity
        weights = [1.0, 1.0]
        result = power_mean(values, weights, p=0.5, epsilon=0.0)
        # [(0.64^0.5 + 0.81^0.5) / 2]^2 = [(0.8 + 0.9) / 2]^2 = 0.85^2
        expected = ((0.64**0.5 + 0.81**0.5) / 2.0) ** 2.0
        assert result == pytest.approx(expected, abs=1e-5)

    def test_weighted_power_mean(self):
        """Test non-uniform weights."""
        values = [0.5, 1.0]
        weights = [2.0, 1.0]
        result = power_mean(values, weights, p=1.0, epsilon=0.0)
        # Weighted arithmetic: (2*0.5 + 1*1.0) / 3 = 2/3
        expected = (2.0 * 0.5 + 1.0 * 1.0) / 3.0
        assert result == pytest.approx(expected, abs=1e-5)

    def test_epsilon_smoothing(self):
        """Test epsilon prevents zero issues."""
        values = [0.0, 0.5]
        weights = [1.0, 1.0]
        epsilon = 0.05
        result = power_mean(values, weights, p=0.5, epsilon=epsilon)
        # [(0.05^0.5 + 0.55^0.5) / 2]^2
        smoothed = [v + epsilon for v in values]
        expected = ((smoothed[0]**0.5 + smoothed[1]**0.5) / 2.0) ** 2.0
        assert result == pytest.approx(expected, abs=1e-5)

    def test_empty_inputs(self):
        """Test empty inputs return 0."""
        assert power_mean([], [], p=0.5) == 0.0

    def test_mismatched_lengths(self):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            power_mean([0.5, 0.6], [1.0], p=0.5)

    def test_zero_weight_sum(self):
        """Test zero weight sum returns 0."""
        values = [0.5, 0.6]
        weights = [0.0, 0.0]
        result = power_mean(values, weights, p=0.5)
        assert result == 0.0


class TestComputeLRS:
    """Tests for compute_lrs function."""

    def test_manual_computation(self):
        """Test LRS computation with manual formula verification."""
        # Simple inputs for hand calculation
        coh_plus = 0.8
        coh_minus = 0.7
        opposition = 0.6
        orthogonality = 0.5
        confound_independence = 0.9
        predictivity_gain = 0.4  # Will be normalized to 0.7

        p = 0.5
        epsilon = 0.05

        result = compute_lrs(
            coh_plus=coh_plus,
            coh_minus=coh_minus,
            opposition=opposition,
            orthogonality=orthogonality,
            confound_independence=confound_independence,
            predictivity_gain=predictivity_gain,
            p=p,
            epsilon=epsilon,
        )

        # Manually compute
        pred_norm = (1.0 + predictivity_gain) / 2.0
        values = [coh_plus, coh_minus, opposition, orthogonality,
                  confound_independence, pred_norm]
        weights = [1.0] * 6
        expected = power_mean(values, weights, p=p, epsilon=epsilon)

        assert result.lrs == pytest.approx(expected, abs=1e-5)
        assert result.breakdown.predictivity_gain_norm == pytest.approx(pred_norm, abs=1e-5)
        assert result.p == p
        assert result.epsilon == epsilon

    def test_predictivity_gain_normalization(self):
        """Test predictivity_gain is normalized correctly."""
        result = compute_lrs(
            coh_plus=0.5,
            coh_minus=0.5,
            opposition=0.5,
            orthogonality=0.5,
            confound_independence=0.5,
            predictivity_gain=0.6,  # (1 + 0.6) / 2 = 0.8
        )
        assert result.breakdown.predictivity_gain_norm == pytest.approx(0.8, abs=1e-5)

    def test_negative_predictivity_gain(self):
        """Test negative predictivity_gain normalized correctly."""
        result = compute_lrs(
            coh_plus=0.5,
            coh_minus=0.5,
            opposition=0.5,
            orthogonality=0.5,
            confound_independence=0.5,
            predictivity_gain=-0.4,  # (1 - 0.4) / 2 = 0.3
        )
        assert result.breakdown.predictivity_gain_norm == pytest.approx(0.3, abs=1e-5)

    def test_custom_weights(self):
        """Test custom component weights."""
        weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        result = compute_lrs(
            coh_plus=0.8,
            coh_minus=0.7,
            opposition=0.5,
            orthogonality=0.5,
            confound_independence=0.5,
            predictivity_gain=0.0,
            weights=weights,
        )
        assert result.breakdown.weights == weights

    def test_all_components_equal(self):
        """Test when all components are equal."""
        value = 0.6
        result = compute_lrs(
            coh_plus=value,
            coh_minus=value,
            opposition=value,
            orthogonality=value,
            confound_independence=value,
            predictivity_gain=2*value - 1,  # Normalized to 0.6
            epsilon=0.0,
        )
        # With uniform values and weights, power mean equals the value
        assert result.lrs == pytest.approx(value, abs=1e-5)

    def test_breakdown_stored_correctly(self):
        """Test breakdown contains all components."""
        result = compute_lrs(
            coh_plus=0.9,
            coh_minus=0.8,
            opposition=0.7,
            orthogonality=0.6,
            confound_independence=0.5,
            predictivity_gain=0.4,
        )
        breakdown = result.breakdown
        assert breakdown.coh_plus == 0.9
        assert breakdown.coh_minus == 0.8
        assert breakdown.opposition == 0.7
        assert breakdown.orthogonality == 0.6
        assert breakdown.confound_independence == 0.5
        assert breakdown.predictivity_gain_norm == pytest.approx(0.7, abs=1e-5)


class TestComputeLRSFromBreakdown:
    """Tests for compute_lrs_from_breakdown function."""

    def test_from_breakdown_matches_direct(self):
        """Test computing from breakdown matches direct computation."""
        # First compute directly
        result1 = compute_lrs(
            coh_plus=0.8,
            coh_minus=0.7,
            opposition=0.6,
            orthogonality=0.5,
            confound_independence=0.9,
            predictivity_gain=0.4,
            p=0.5,
            epsilon=0.05,
        )

        # Then compute from breakdown
        lrs2 = compute_lrs_from_breakdown(
            result1.breakdown,
            p=0.5,
            epsilon=0.05,
        )

        assert lrs2 == pytest.approx(result1.lrs, abs=1e-5)


class TestComputeLRSBatch:
    """Tests for compute_lrs_batch function."""

    def test_batch_computation(self):
        """Test batch computation produces correct results."""
        components_data = {
            "comp1": {
                "coh_plus": 0.8,
                "coh_minus": 0.7,
                "opposition": 0.6,
                "orthogonality": 0.5,
                "confound_independence": 0.9,
                "predictivity_gain": 0.4,
            },
            "comp2": {
                "coh_plus": 0.5,
                "coh_minus": 0.5,
                "opposition": 0.5,
                "orthogonality": 0.5,
                "confound_independence": 0.5,
                "predictivity_gain": 0.0,
            },
        }

        results = compute_lrs_batch(components_data, p=0.5, epsilon=0.05)

        # Verify individual results
        result1 = compute_lrs(**components_data["comp1"], p=0.5, epsilon=0.05)
        result2 = compute_lrs(**components_data["comp2"], p=0.5, epsilon=0.05)

        assert results["comp1"].lrs == pytest.approx(result1.lrs, abs=1e-5)
        assert results["comp2"].lrs == pytest.approx(result2.lrs, abs=1e-5)

    def test_empty_batch(self):
        """Test empty batch returns empty dict."""
        results = compute_lrs_batch({})
        assert results == {}


class TestClassification:
    """Tests for is_strong_representation and is_weak_representation."""

    def test_strong_representation(self):
        """Test strong representation threshold (>0.7)."""
        assert is_strong_representation(0.8) is True
        assert is_strong_representation(0.71) is True
        assert is_strong_representation(0.7) is False
        assert is_strong_representation(0.69) is False

    def test_weak_representation(self):
        """Test weak representation threshold (<0.4)."""
        assert is_weak_representation(0.3) is True
        assert is_weak_representation(0.39) is True
        assert is_weak_representation(0.4) is False
        assert is_weak_representation(0.41) is False

    def test_moderate_representation(self):
        """Test moderate range (neither strong nor weak)."""
        lrs = 0.55
        assert not is_strong_representation(lrs)
        assert not is_weak_representation(lrs)

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        assert is_strong_representation(0.65, threshold=0.6) is True
        assert is_weak_representation(0.35, threshold=0.3) is False


class TestLRSIntegration:
    """Integration tests for LRS computation."""

    def test_high_quality_components(self):
        """Test high-quality components yield strong LRS."""
        result = compute_lrs(
            coh_plus=0.9,
            coh_minus=0.85,
            opposition=0.8,
            orthogonality=0.75,
            confound_independence=0.95,
            predictivity_gain=0.7,
        )
        assert is_strong_representation(result.lrs)

    def test_low_quality_components(self):
        """Test low-quality components yield weak LRS."""
        result = compute_lrs(
            coh_plus=0.2,
            coh_minus=0.25,
            opposition=0.3,
            orthogonality=0.2,
            confound_independence=0.15,
            predictivity_gain=-0.5,
        )
        assert is_weak_representation(result.lrs)

    def test_lrs_result_dataclass(self):
        """Test LRSResult dataclass structure."""
        breakdown = LRSBreakdown(
            coh_plus=0.8,
            coh_minus=0.7,
            opposition=0.6,
            orthogonality=0.5,
            confound_independence=0.9,
            predictivity_gain_norm=0.7,
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = LRSResult(
            lrs=0.72,
            breakdown=breakdown,
            p=0.5,
            epsilon=0.05,
        )
        assert result.lrs == 0.72
        assert result.p == 0.5
        assert result.epsilon == 0.05
        assert result.breakdown.coh_plus == 0.8
