"""Tests for projection-level weight domain (SS4).

Covers: RMS normalization (Def 4.2), sig/sig_bar (Def 2.3),
opposition degree (Def 3.4), confidence (Def 4.6-4.7),
aggregation (Def 3.3), and SVD diagnostics (Def 4.5).
"""

import pytest
import torch

from dcaf.domains.weight.delta import compute_projection_rms, compute_base_relative_delta
from dcaf.domains.weight.significance import sig, sig_bar, compute_significance, compute_baseline_insignificance
from dcaf.domains.weight.opposition import compute_opposition_degree, is_bidirectional
from dcaf.domains.weight.aggregation import compute_cluster_delta_matrix, compute_cluster_deltas
from dcaf.domains.weight.confidence import compute_projection_confidence, aggregate_component_confidence
from dcaf.domains.weight.svd import compute_svd_diagnostics


class TestProjectionRMS:
    """Def 4.2: ||dW||_RMS = ||dW||_F / sqrt(m*n)."""

    def test_zero_delta(self):
        delta = torch.zeros(64, 32)
        assert compute_projection_rms(delta) == 0.0

    def test_ones_delta(self):
        delta = torch.ones(64, 32)
        rms = compute_projection_rms(delta)
        assert abs(rms - 1.0) < 1e-6

    def test_scaling(self):
        torch.manual_seed(42)
        delta = torch.randn(64, 32)
        rms1 = compute_projection_rms(delta)
        rms2 = compute_projection_rms(delta * 3.0)
        assert abs(rms2 / rms1 - 3.0) < 1e-5

    def test_shape_normalization(self):
        torch.manual_seed(42)
        small = torch.randn(16, 8) * 0.1
        large = torch.randn(256, 128) * 0.1
        rms_small = compute_projection_rms(small)
        rms_large = compute_projection_rms(large)
        assert abs(rms_small - rms_large) < 0.05, "RMS should be similar for same element scale"


class TestSignificancePredicates:
    """Def 2.3: sig and sig_bar on projection RMS norms."""

    def setup_method(self):
        torch.manual_seed(42)
        self.rms_values = {f"proj_{i}": abs(torch.randn(1).item()) for i in range(100)}

    def test_sig_top_15_percent(self):
        result = compute_significance(self.rms_values, tau_sig=85)
        sig_count = sum(result.values())
        assert 10 <= sig_count <= 20, f"Expected ~15 significant, got {sig_count}"

    def test_sig_bar_bottom_50_percent(self):
        result = compute_baseline_insignificance(self.rms_values, tau_base=50)
        bar_count = sum(result.values())
        assert 45 <= bar_count <= 55, f"Expected ~50 insignificant, got {bar_count}"

    def test_sig_single_projection(self):
        rms = {"a": 0.1, "b": 0.5, "c": 0.9}
        assert sig("c", rms) is True
        assert sig("a", rms) is False

    def test_sig_bar_single_projection(self):
        rms = {"a": 0.1, "b": 0.5, "c": 0.9}
        assert sig_bar("a", rms) is True
        assert sig_bar("c", rms) is False

    def test_sig_and_sig_bar_exclusive(self):
        rms = {f"p{i}": float(i) for i in range(100)}
        for name in rms:
            s = sig(name, rms)
            sb = sig_bar(name, rms)
            assert not (s and sb), f"sig and sig_bar should not both be True for {name}"


class TestOppositionDegree:
    """Def 3.4: cosine opposition on flattened matrices."""

    def test_perfectly_opposing(self):
        delta_plus = torch.ones(32, 16)
        delta_minus = -torch.ones(32, 16)
        cos_opp, opp_deg = compute_opposition_degree(delta_plus, delta_minus)
        assert abs(cos_opp - (-1.0)) < 1e-5
        assert abs(opp_deg - 1.0) < 1e-5

    def test_perfectly_aligned(self):
        delta = torch.ones(32, 16)
        cos_opp, opp_deg = compute_opposition_degree(delta, delta)
        assert abs(cos_opp - 1.0) < 1e-5
        assert opp_deg == 0.0

    def test_orthogonal(self):
        a = torch.zeros(4, 4)
        a[0, 0] = 1.0
        b = torch.zeros(4, 4)
        b[1, 1] = 1.0
        cos_opp, opp_deg = compute_opposition_degree(a, b)
        assert abs(cos_opp) < 1e-5
        assert opp_deg == 0.0

    def test_zero_delta_handled(self):
        z = torch.zeros(16, 8)
        nz = torch.randn(16, 8)
        cos_opp, opp_deg = compute_opposition_degree(z, nz)
        assert opp_deg == 0.0

    def test_bidirectional_threshold(self):
        assert is_bidirectional(0.5) is True
        assert is_bidirectional(0.2) is False
        assert is_bidirectional(0.3) is False


class TestClusterAggregation:
    """Def 3.3: effectiveness-weighted cluster deltas."""

    def test_single_signal(self):
        torch.manual_seed(42)
        delta = torch.randn(32, 16)
        deltas_by_signal = {"t1": {"proj_A": delta}}
        effectiveness = {"t1": 1.0}
        result = compute_cluster_delta_matrix("proj_A", deltas_by_signal, effectiveness, ["t1"])
        assert torch.allclose(result, delta, atol=1e-6)

    def test_effectiveness_weighting(self):
        torch.manual_seed(42)
        d1 = torch.ones(8, 4)
        d2 = torch.ones(8, 4) * 3.0
        deltas = {"t1": {"p": d1}, "t2": {"p": d2}}
        eff = {"t1": 0.0, "t2": 1.0}
        result = compute_cluster_delta_matrix("p", deltas, eff, ["t1", "t2"])
        assert torch.allclose(result, d2, atol=1e-5), "Zero-eff signal should be ignored"


class TestProjectionConfidence:
    """Def 4.6-4.7: C_W per projection, MAX to component."""

    def test_high_presence_high_opposition(self):
        rms = {"t1": {"p": 0.9}, "t2": {"p": 0.8}, "t6": {"p": 0.7}}
        eff = {"t1": 1.0, "t2": 0.9, "t6": 0.8}
        C_W = compute_projection_confidence(
            "p", rms, eff, opp_degree=0.8,
            behavioral_signals=["t1", "t2", "t6"], baseline_signals=[],
        )
        assert C_W > 0.5, "High presence + high opposition should give high confidence"
        assert C_W <= 1.0

    def test_baseline_filter_zeros_out(self):
        rms = {"t1": {"p": 0.9}, "t11": {"p": 0.9}}
        eff = {"t1": 1.0, "t11": 0.5}
        C_W = compute_projection_confidence(
            "p", rms, eff, opp_degree=0.5,
            behavioral_signals=["t1"], baseline_signals=["t11"],
        )
        assert C_W == 0.0, "Significant in baseline should zero confidence"

    def test_component_aggregation_is_max(self):
        projs = {"L5H3_Q": 0.3, "L5H3_K": 0.1, "L5H3_V": 0.9, "L5H3_O": 0.4}
        result = aggregate_component_confidence(list(projs.keys()), projs)
        assert result == 0.9


class TestSVDDiagnostics:
    """Def 4.5: rank-1 fraction and spectral opposition."""

    def test_rank1_delta(self):
        u = torch.randn(32, 1)
        v = torch.randn(1, 16)
        rank1_matrix = u @ v
        diag = compute_svd_diagnostics(rank1_matrix, rank1_matrix * -1)
        assert diag.rank_1_fraction > 0.95, "Rank-1 matrix should have high rank-1 fraction"

    def test_spectral_opposition_opposing_deltas(self):
        torch.manual_seed(42)
        delta = torch.randn(32, 16)
        diag = compute_svd_diagnostics(delta, -delta)
        assert diag.spectral_opposition > 0.8, "Opposite deltas should have high spectral opposition"
