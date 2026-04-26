"""Test edge pathway attribution (Def 9.8)."""

import torch

from dcaf.circuit.pathway import (
    PathwayAttribution,
    compute_pathway_attribution,
    compute_pathway_from_weight_deltas,
)


class TestPathwayAttribution:
    def test_v_dominant(self):
        dQ = torch.randn(64, 32) * 0.1
        dK = torch.randn(64, 32) * 0.1
        dV = torch.randn(64, 32) * 1.0
        result = compute_pathway_attribution(dQ, dK, dV, "L5_MLP", "L8H3")
        assert result.dominant_pathway == "V"
        assert result.w_V > result.w_Q
        assert result.w_V > result.w_K
        assert result.is_attention_target is True

    def test_weights_sum_to_one(self):
        dQ = torch.randn(64, 32)
        dK = torch.randn(64, 32)
        dV = torch.randn(64, 32)
        result = compute_pathway_attribution(dQ, dK, dV, "L5_MLP", "L8H3")
        total = result.w_Q + result.w_K + result.w_V
        assert abs(total - 1.0) < 1e-6

    def test_zero_deltas_returns_residual(self):
        z = torch.zeros(32, 16)
        result = compute_pathway_attribution(z, z, z, "L5_MLP", "L8H3")
        assert result.dominant_pathway == "residual"
        assert result.is_attention_target is False

    def test_via_property_attention(self):
        dQ = torch.randn(32, 16) * 0.5
        dK = torch.randn(32, 16) * 0.3
        dV = torch.randn(32, 16) * 0.8
        result = compute_pathway_attribution(dQ, dK, dV, "L3H1", "L7H2")
        via = result.via
        assert "Q" in via and "K" in via and "V" in via
        assert abs(sum(via.values()) - 1.0) < 1e-6

    def test_via_property_non_attention(self):
        z = torch.zeros(32, 16)
        result = compute_pathway_attribution(z, z, z, "L3H1", "L7_MLP")
        assert result.via == {"residual": 1.0}

    def test_to_dict(self):
        dQ = torch.randn(32, 16)
        dK = torch.randn(32, 16)
        dV = torch.randn(32, 16)
        result = compute_pathway_attribution(dQ, dK, dV, "L3H1", "L7H2")
        d = result.to_dict()
        assert "source" in d and "target" in d and "via" in d

    def test_weight_delta_method(self):
        dQ = torch.randn(64, 32) * 0.1
        dK = torch.randn(64, 32) * 0.1
        dV = torch.randn(64, 32) * 2.0
        result = compute_pathway_from_weight_deltas(dQ, dK, dV, "L5_MLP", "L8H3")
        assert result.dominant_pathway == "V"
        assert result.w_V > 0.5

    def test_q_dominant(self):
        dQ = torch.randn(64, 32) * 5.0
        dK = torch.randn(64, 32) * 0.1
        dV = torch.randn(64, 32) * 0.1
        result = compute_pathway_attribution(dQ, dK, dV, "L3H1", "L7H2")
        assert result.dominant_pathway == "Q"
