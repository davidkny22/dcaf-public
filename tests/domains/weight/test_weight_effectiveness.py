"""
Tests for dcaf.domains.weight.effectiveness module.

All tests run on CPU with synthetic data using torch.manual_seed(42) for reproducibility.
Small tensor dimensions (16-64) for speed. pytest.approx for float comparisons.
"""

import pytest
import torch

from dcaf.domains.weight.effectiveness import (
    SignalType,
    SignalMetrics,
    EffectivenessConfig,
    compute_delta_improve_sft,
    compute_delta_improve_pref_opt,
    compute_effectiveness_raw,
    normalize_effectiveness,
    get_signal_type_from_name,
)
from dcaf.core.defaults import BETA, EPS_RMS


# ============================================================================
# TestDeltaImproveSft
# ============================================================================


class TestDeltaImproveSft:
    """Test SFT improvement delta: (pre - post) / (|pre| + eps)"""

    def test_sft_loss_decrease_positive(self):
        """When loss decreases, delta_improve > 0 (improvement)."""
        torch.manual_seed(42)
        pre_loss = 2.0
        post_loss = 1.0

        delta = compute_delta_improve_sft(pre_loss, post_loss)

        # Expected: (2.0 - 1.0) / (2.0 + eps) ≈ 0.5
        expected = (pre_loss - post_loss) / (abs(pre_loss) + EPS_RMS)
        assert delta == pytest.approx(expected)
        assert delta > 0

    def test_sft_loss_increase_negative(self):
        """When loss increases, delta_improve < 0 (worse after training)."""
        torch.manual_seed(42)
        pre_loss = 1.0
        post_loss = 2.0

        delta = compute_delta_improve_sft(pre_loss, post_loss)

        # Expected: (1.0 - 2.0) / (1.0 + eps) ≈ -1.0
        expected = (pre_loss - post_loss) / (abs(pre_loss) + EPS_RMS)
        assert delta == pytest.approx(expected)
        assert delta < 0

    def test_sft_zero_pre_loss_returns_zero(self):
        """When pre_loss near zero, division guard returns 0.0."""
        torch.manual_seed(42)
        pre_loss = 0.0
        post_loss = 1.0

        delta = compute_delta_improve_sft(pre_loss, post_loss, eps=1e-8)

        assert delta == 0.0


# ============================================================================
# TestDeltaImprovePrefOpt
# ============================================================================


class TestDeltaImprovePrefOpt:
    """Test PrefOpt improvement delta: (post - pre) / (|pre| + eps)"""

    def test_pref_opt_margin_increase(self):
        """When margin increases, delta_improve > 0."""
        torch.manual_seed(42)
        pre_margin = 0.5
        post_margin = 1.0

        delta = compute_delta_improve_pref_opt(pre_margin, post_margin)

        # Expected: (1.0 - 0.5) / (0.5 + eps) ≈ 1.0
        expected = (post_margin - pre_margin) / (abs(pre_margin) + EPS_RMS)
        assert delta == pytest.approx(expected)
        assert delta > 0

    def test_pref_opt_margin_decrease(self):
        """When margin decreases, delta_improve < 0."""
        torch.manual_seed(42)
        pre_margin = 1.0
        post_margin = 0.5

        delta = compute_delta_improve_pref_opt(pre_margin, post_margin)

        # Expected: (0.5 - 1.0) / (1.0 + eps) ≈ -0.5
        expected = (post_margin - pre_margin) / (abs(pre_margin) + EPS_RMS)
        assert delta == pytest.approx(expected)
        assert delta < 0

    def test_pref_opt_zero_pre_margin(self):
        """When pre_margin is zero, eps prevents division by zero."""
        torch.manual_seed(42)
        pre_margin = 0.0
        post_margin = 0.5
        eps = 1e-8

        delta = compute_delta_improve_pref_opt(pre_margin, post_margin, eps=eps)

        # Expected: (0.5 - 0.0) / (0.0 + 1e-8) = 0.5 / 1e-8 (large positive)
        expected = (post_margin - pre_margin) / (abs(pre_margin) + eps)
        assert delta == pytest.approx(expected)
        assert delta > 0


# ============================================================================
# TestEffectivenessRaw
# ============================================================================


class TestEffectivenessRaw:
    """Test raw effectiveness computation: delta_improve + beta*threshold"""

    def test_sft_with_threshold_bonus(self):
        """SFT signal with threshold crossed gets beta bonus."""
        torch.manual_seed(42)
        metrics = SignalMetrics(
            signal_name="test_sft",
            signal_type=SignalType.SFT,
            pre_loss=2.0,
            post_loss=1.0,
            crossed_threshold=True,
        )
        config = EffectivenessConfig(beta=0.1, eps=1e-8)

        eff_raw = compute_effectiveness_raw(metrics, config)

        # Expected: delta_improve + 0.1
        delta = (2.0 - 1.0) / (2.0 + 1e-8)
        expected = delta + 0.1
        assert eff_raw == pytest.approx(expected)

    def test_sft_without_threshold_bonus(self):
        """SFT signal without threshold gets no bonus."""
        torch.manual_seed(42)
        metrics = SignalMetrics(
            signal_name="test_sft",
            signal_type=SignalType.SFT,
            pre_loss=2.0,
            post_loss=1.0,
            crossed_threshold=False,
        )
        config = EffectivenessConfig(beta=0.1, eps=1e-8)

        eff_raw = compute_effectiveness_raw(metrics, config)

        # Expected: delta_improve + 0
        delta = (2.0 - 1.0) / (2.0 + 1e-8)
        expected = delta
        assert eff_raw == pytest.approx(expected)

    def test_pref_opt_signal(self):
        """PrefOpt signal uses margin metrics."""
        torch.manual_seed(42)
        metrics = SignalMetrics(
            signal_name="test_simpo",
            signal_type=SignalType.PREF_OPT,
            pre_margin=0.5,
            post_margin=1.0,
            crossed_threshold=True,
        )
        config = EffectivenessConfig(beta=0.1, eps=1e-8)

        eff_raw = compute_effectiveness_raw(metrics, config)

        # Expected: (1.0 - 0.5) / (0.5 + eps) + 0.1
        delta = (1.0 - 0.5) / (0.5 + 1e-8)
        expected = delta + 0.1
        assert eff_raw == pytest.approx(expected)

    def test_anti_signal_type(self):
        """Anti signals use loss metrics."""
        torch.manual_seed(42)
        metrics = SignalMetrics(
            signal_name="anti_test",
            signal_type=SignalType.ANTI,
            pre_loss=2.0,
            post_loss=1.0,
            crossed_threshold=False,
        )
        config = EffectivenessConfig(eps=1e-8)

        eff_raw = compute_effectiveness_raw(metrics, config)

        # Expected: (2.0 - 1.0) / (2.0 + eps)
        delta = (2.0 - 1.0) / (2.0 + 1e-8)
        expected = delta
        assert eff_raw == pytest.approx(expected)

    def test_negated_signal_type(self):
        """Negated signals use loss metrics."""
        torch.manual_seed(42)
        metrics = SignalMetrics(
            signal_name="negated_test",
            signal_type=SignalType.NEGATED,
            pre_loss=3.0,
            post_loss=2.0,
            crossed_threshold=True,
        )
        config = EffectivenessConfig(beta=0.2, eps=1e-8)

        eff_raw = compute_effectiveness_raw(metrics, config)

        # Expected: (3.0 - 2.0) / (3.0 + eps) + 0.2
        delta = (3.0 - 2.0) / (3.0 + 1e-8)
        expected = delta + 0.2
        assert eff_raw == pytest.approx(expected)


# ============================================================================
# TestNormalizeEffectiveness
# ============================================================================


class TestNormalizeEffectiveness:
    """Test effectiveness normalization to [0, 1]."""

    def test_normalize_to_01_range(self):
        """Normalized scores are in [0, 1] range."""
        torch.manual_seed(42)
        raw_scores = {
            "signal_a": 0.1,
            "signal_b": 0.5,
            "signal_c": 0.9,
        }

        normalized = normalize_effectiveness(raw_scores)

        for name, score in normalized.items():
            assert 0.0 <= score <= 1.0

    def test_95th_percentile_clipping(self):
        """Values above 95th percentile are clipped."""
        torch.manual_seed(42)
        # Create 100 scores: 0.0 to 0.99
        raw_scores = {f"signal_{i}": i / 100.0 for i in range(100)}
        config = EffectivenessConfig(clip_percentile=95.0)

        normalized = normalize_effectiveness(raw_scores, config)

        # All scores should be in [0, 1] after clipping and normalization
        for name, score in normalized.items():
            assert 0.0 <= score <= 1.0

    def test_single_value(self):
        """Single value returns 0.5."""
        torch.manual_seed(42)
        raw_scores = {"signal_a": 0.7}

        normalized = normalize_effectiveness(raw_scores)

        assert normalized["signal_a"] == 0.5

    def test_all_same_values(self):
        """All same values return 0.5."""
        torch.manual_seed(42)
        raw_scores = {
            "signal_a": 0.5,
            "signal_b": 0.5,
            "signal_c": 0.5,
        }

        normalized = normalize_effectiveness(raw_scores)

        for name, score in normalized.items():
            assert score == 0.5

    def test_empty_returns_empty(self):
        """Empty input returns empty dict."""
        torch.manual_seed(42)
        raw_scores = {}

        normalized = normalize_effectiveness(raw_scores)

        assert normalized == {}


# ============================================================================
# TestSignalTypeInference
# ============================================================================


class TestSignalTypeInference:
    """Test signal type inference from signal names."""

    def test_simpo_inferred_as_pref_opt(self):
        """Signal names with _simpo are inferred as PrefOpt."""
        torch.manual_seed(42)
        signal_name = "delta_t1_prefopt_target"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.PREF_OPT

    def test_sft_inferred_as_sft(self):
        """Signal names with _sft are inferred as SFT."""
        torch.manual_seed(42)
        signal_name = "delta_safe_sft"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.SFT

    def test_anti_inferred_as_anti(self):
        """Signal names with anti_ are inferred as Anti."""
        torch.manual_seed(42)
        signal_name = "anti_delta_safe"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.ANTI

    def test_negated_inferred_as_negated(self):
        """Signal names with negated_ are inferred as Negated."""
        torch.manual_seed(42)
        signal_name = "negated_delta_safe"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.NEGATED

    def test_language_inferred_as_language(self):
        """Signal names with language are inferred as Language."""
        torch.manual_seed(42)
        signal_name = "delta_t11_baseline"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.LANGUAGE

    def test_unknown_defaults_to_sft(self):
        """Unknown signal names default to SFT."""
        torch.manual_seed(42)
        signal_name = "some_totally_unknown_signal"

        signal_type = get_signal_type_from_name(signal_name)

        assert signal_type == SignalType.SFT
