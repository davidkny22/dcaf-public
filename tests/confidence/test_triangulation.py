"""
Tests for confidence triangulation from multiple measurement domains.

Tests cover the triangulation formula:
C_base = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^{1/(w+2)}

And unified confidence with multi-path bonus:
C_unified = min(1, C_base + β_path * max(0, path_count - 1))
"""

import pytest
import torch
from dcaf.confidence.triangulation import (
    triangulate,
    TriangulationConfig,
    UnifiedConfidence,
    compute_unified_confidence,
    compute_domain_contribution,
    compute_domain_deviations,
    compute_domain_disagreement,
)
from dcaf.domains.base import TriangulatedConfidence


# Set random seed for reproducibility
torch.manual_seed(42)


class TestTriangulate:
    """Tests for triangulate function."""

    def test_formula_manual_computation(self):
        """Test triangulation formula with manual computation."""
        C_W = 0.6
        C_A = 0.7
        C_G = 0.5
        w = 2  # weight_power
        eps = 0.05

        result = triangulate(C_W, C_A, C_G)

        # Manual computation: [(C_W + eps)^w * (C_A + eps) * (C_G + eps)]^(1/(w+2))
        product = (C_W + eps)**w * (C_A + eps) * (C_G + eps)
        expected = product ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)
        assert result.C_W.value == C_W
        assert result.C_A.value == C_A
        assert result.C_G.value == C_G

    def test_all_domains_equal(self):
        """Test when all domains have equal confidence."""
        value = 0.6
        result = triangulate(C_W=value, C_A=value, C_G=value)

        # With epsilon=0.05, w=2: [(0.65)^2 * 0.65 * 0.65]^(1/4)
        eps = 0.05
        w = 2
        expected = ((value + eps)**w * (value + eps) * (value + eps)) ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)

    def test_weight_domain_emphasized(self):
        """Test weight domain has w=2 power (emphasized)."""
        # Weight domain higher, should pull result up more due to w=2 power
        C_W = 0.8
        C_A = 0.5
        C_G = 0.5

        result = triangulate(C_W, C_A, C_G)

        # Compare to if all had equal power
        eps = 0.05
        w = 2
        actual_formula = ((C_W + eps)**w * (C_A + eps) * (C_G + eps)) ** (1.0 / (w + 2))
        equal_power = ((C_W + eps) * (C_A + eps) * (C_G + eps)) ** (1.0 / 3)

        assert result.value == pytest.approx(actual_formula, abs=1e-5)
        # Weight emphasis should make result higher than equal power
        assert result.value > equal_power

    def test_missing_domain_uses_default(self):
        """Test missing domain uses default value (0.5)."""
        C_W = 0.7
        C_A = 0.6
        # C_G is missing, should default to 0.5

        result = triangulate(C_W=C_W, C_A=C_A, C_G=None)

        eps = 0.05
        w = 2
        C_G_default = 0.5  # DEFAULT_MISSING_CONFIDENCE
        expected = ((C_W + eps)**w * (C_A + eps) * (C_G_default + eps)) ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)

    def test_all_missing_domains(self):
        """Test all domains missing uses neutral defaults."""
        result = triangulate(C_W=None, C_A=None, C_G=None)

        # All default to 0.5
        eps = 0.05
        w = 2
        default = 0.5
        expected = ((default + eps)**w * (default + eps) * (default + eps)) ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)

    def test_custom_config(self):
        """Test custom triangulation configuration."""
        config = TriangulationConfig(
            weight_power=3,  # Higher weight emphasis
            epsilon=0.1,  # Larger smoothing
        )

        C_W = 0.6
        C_A = 0.7
        C_G = 0.5

        result = triangulate(C_W, C_A, C_G, config=config)

        # Manual with custom config
        w = 3
        eps = 0.1
        expected = ((C_W + eps)**w * (C_A + eps) * (C_G + eps)) ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)

    def test_require_all_domains(self):
        """Test require_all_domains=True returns None if any missing."""
        config = TriangulationConfig(require_all_domains=True)

        # All present - should succeed
        result = triangulate(C_W=0.6, C_A=0.7, C_G=0.5, config=config)
        assert result is not None

        # One missing - should fail
        result = triangulate(C_W=0.6, C_A=0.7, C_G=None, config=config)
        assert result is None

    def test_epsilon_prevents_zero(self):
        """Test epsilon prevents zero values in formula."""
        # Without epsilon, zero would break the formula
        C_W = 0.0
        C_A = 0.0
        C_G = 0.0

        result = triangulate(C_W, C_A, C_G)

        # With epsilon=0.05: [(0.05)^2 * 0.05 * 0.05]^(1/4)
        eps = 0.05
        w = 2
        expected = (eps**w * eps * eps) ** (1.0 / (w + 2))

        assert result.value == pytest.approx(expected, abs=1e-5)
        assert result.value > 0  # Not zero

    def test_clamped_to_one(self):
        """Test result clamped to [0, 1] range."""
        # Even with high values, should not exceed 1
        C_W = 1.0
        C_A = 1.0
        C_G = 1.0

        result = triangulate(C_W, C_A, C_G)

        assert 0.0 <= result.value <= 1.0


class TestMultiPathBonus:
    """Tests for unified confidence with multi-path bonus."""

    def test_single_path_no_bonus(self):
        """Test single path (path_count=1) has no bonus."""
        C_W = 0.6
        C_A = 0.7
        C_G = 0.5

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=1,
            beta_path=0.15,
        )

        # bonus = 0.15 * max(0, 1 - 1) = 0
        assert unified.bonus == 0.0
        assert unified.path_count == 1

        # Compute base triangulated
        base = triangulate(C_W, C_A, C_G)
        assert unified.C_base == pytest.approx(base.value, abs=1e-5)
        assert unified.value == pytest.approx(base.value, abs=1e-5)

    def test_two_paths_bonus(self):
        """Test two paths (path_count=2) adds bonus of 0.15."""
        C_W = 0.6
        C_A = 0.7
        C_G = 0.5
        beta_path = 0.15

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=2,
            beta_path=beta_path,
        )

        # bonus = 0.15 * max(0, 2 - 1) = 0.15
        expected_bonus = beta_path * 1
        assert unified.bonus == pytest.approx(expected_bonus, abs=1e-5)
        assert unified.path_count == 2

        # Compute base triangulated
        base = triangulate(C_W, C_A, C_G)
        expected_value = base.value + expected_bonus

        assert unified.C_base == pytest.approx(base.value, abs=1e-5)
        assert unified.value == pytest.approx(expected_value, abs=1e-5)

    def test_three_paths_max_bonus(self):
        """Test three paths (path_count=3) adds bonus of 0.30."""
        C_W = 0.6
        C_A = 0.7
        C_G = 0.5
        beta_path = 0.15

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=3,
            beta_path=beta_path,
        )

        # bonus = 0.15 * max(0, 3 - 1) = 0.30
        expected_bonus = beta_path * 2
        assert unified.bonus == pytest.approx(expected_bonus, abs=1e-5)
        assert unified.path_count == 3

        # Compute base triangulated
        base = triangulate(C_W, C_A, C_G)
        expected_value = base.value + expected_bonus

        assert unified.C_base == pytest.approx(base.value, abs=1e-5)
        assert unified.value == pytest.approx(expected_value, abs=1e-5)

    def test_unified_capped_at_one(self):
        """Test unified confidence capped at 1.0."""
        # High base + bonus should cap at 1.0
        C_W = 0.9
        C_A = 0.9
        C_G = 0.9

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=3,
            beta_path=0.15,
        )

        # Even if C_base + bonus > 1, value should be clamped
        assert unified.value <= 1.0
        assert unified.value == pytest.approx(1.0, abs=1e-5)

    def test_custom_beta_path(self):
        """Test custom beta_path parameter."""
        C_W = 0.5
        C_A = 0.5
        C_G = 0.5
        beta_path = 0.25  # Custom bonus weight

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=2,
            beta_path=beta_path,
        )

        expected_bonus = beta_path * 1
        assert unified.bonus == pytest.approx(expected_bonus, abs=1e-5)

    def test_zero_path_count(self):
        """Test path_count=0 (edge case, should have no bonus)."""
        C_W = 0.6
        C_A = 0.6
        C_G = 0.6

        unified = compute_unified_confidence(
            C_W=C_W, C_A=C_A, C_G=C_G,
            path_count=0,
            beta_path=0.15,
        )

        # bonus = 0.15 * max(0, 0 - 1) = 0
        assert unified.bonus == 0.0


class TestDomainDiagnostics:
    """Tests for domain contribution and disagreement analysis."""

    def test_domain_contribution(self):
        """Test relative contribution calculation."""
        C_W = 0.8
        C_A = 0.6
        C_G = 0.4

        contrib = compute_domain_contribution(C_W, C_A, C_G)

        # Weight should have highest contribution (w=2 power)
        assert contrib["weight"] > contrib["activation"]
        assert contrib["weight"] > contrib["geometry"]

        # Contributions should sum to 1.0
        total = contrib["weight"] + contrib["activation"] + contrib["geometry"]
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_domain_deviations(self):
        """Test deviation from mean."""
        C_W = 0.8
        C_A = 0.6
        C_G = 0.4

        deviations = compute_domain_deviations(C_W, C_A, C_G)

        mean = (C_W + C_A + C_G) / 3.0

        assert deviations["weight"] == pytest.approx(C_W - mean, abs=1e-5)
        assert deviations["activation"] == pytest.approx(C_A - mean, abs=1e-5)
        assert deviations["geometry"] == pytest.approx(C_G - mean, abs=1e-5)

        # Deviations sum to zero
        total_dev = deviations["weight"] + deviations["activation"] + deviations["geometry"]
        assert total_dev == pytest.approx(0.0, abs=1e-5)

    def test_domain_disagreement_low(self):
        """Test low disagreement (convergent evidence)."""
        # All domains close together
        C_W = 0.60
        C_A = 0.62
        C_G = 0.58

        disagreement = compute_domain_disagreement(C_W, C_A, C_G)

        # Should be low variance
        assert disagreement < 0.02

    def test_domain_disagreement_high(self):
        """Test high disagreement (conflicting signals)."""
        # Domains far apart
        C_W = 0.9
        C_A = 0.5
        C_G = 0.2

        disagreement = compute_domain_disagreement(C_W, C_A, C_G)

        # Should be high variance
        assert disagreement > 0.08

    def test_domain_disagreement_zero(self):
        """Test zero disagreement (all equal)."""
        C_W = 0.7
        C_A = 0.7
        C_G = 0.7

        disagreement = compute_domain_disagreement(C_W, C_A, C_G)

        assert disagreement == pytest.approx(0.0, abs=1e-5)


class TestTriangulationIntegration:
    """Integration tests for triangulation."""

    def test_high_confidence_all_domains(self):
        """Test high confidence in all domains."""
        result = triangulate(C_W=0.9, C_A=0.85, C_G=0.8)
        assert result.value > 0.7  # Should be strong confidence

    def test_low_confidence_all_domains(self):
        """Test low confidence in all domains."""
        result = triangulate(C_W=0.2, C_A=0.25, C_G=0.3)
        assert result.value < 0.4  # Should be weak confidence

    def test_mixed_domain_confidences(self):
        """Test mixed domain confidences."""
        # High weight, medium others
        result = triangulate(C_W=0.9, C_A=0.5, C_G=0.5)

        # Weight emphasis should pull result above simple average
        simple_avg = (0.9 + 0.5 + 0.5) / 3.0
        assert result.value > simple_avg

    def test_unified_with_multi_path_boost(self):
        """Test multi-path discovery boosts confidence significantly."""
        C_W = 0.5
        C_A = 0.5
        C_G = 0.5

        # Single path
        unified_1 = compute_unified_confidence(C_W=C_W, C_A=C_A, C_G=C_G, path_count=1)

        # Three paths
        unified_3 = compute_unified_confidence(C_W=C_W, C_A=C_A, C_G=C_G, path_count=3)

        # Three paths should be 0.30 higher
        assert unified_3.value == pytest.approx(unified_1.value + 0.30, abs=1e-5)

    def test_unified_confidence_dataclass(self):
        """Test UnifiedConfidence dataclass structure."""
        unified = UnifiedConfidence.compute(
            C_W=0.6, C_A=0.7, C_G=0.5,
            path_count=2,
            beta_path=0.15,
        )

        assert hasattr(unified, 'value')
        assert hasattr(unified, 'C_base')
        assert hasattr(unified, 'bonus')
        assert hasattr(unified, 'path_count')
        assert unified.C_W == 0.6
        assert unified.C_A == 0.7
        assert unified.C_G == 0.5

    def test_triangulated_confidence_repr(self):
        """Test TriangulatedConfidence string representation."""
        result = triangulate(C_W=0.6, C_A=0.7, C_G=0.5)
        repr_str = repr(result)
        assert "TriangulatedConfidence" in repr_str
        assert "C=" in repr_str
