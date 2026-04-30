"""End-to-end smoke tests with a tiny neural network.

Verifies the full mathematical pipeline works on real tensors,
using TinyTransformer from conftest (~50K params, CPU only).
"""

import torch

from dcaf.core.config import DCAFConfig
from dcaf.core.signals import CANONICAL_SIGNALS, get_target_signals, get_opposite_signals
from dcaf.domains.weight.delta import compute_projection_rms
from dcaf.domains.weight.significance import sig, sig_bar
from dcaf.domains.weight.aggregation import compute_cluster_delta_matrix
from dcaf.domains.weight.opposition import compute_opposition_degree
from dcaf.domains.weight.confidence import compute_projection_confidence, aggregate_component_confidence
from dcaf.domains.weight.svd import compute_svd_diagnostics
from dcaf.domains.base import TriangulatedConfidence
from dcaf.confidence.triangulation import UnifiedConfidence
from dcaf.circuit.graph import CircuitGraph, CircuitNode, CircuitEdge
from dcaf.circuit.pathway import compute_pathway_from_weight_deltas
from dcaf.ablation.classification import classify_final, ComponentStatus
from dcaf.ablation.superadditivity import InteractionRequirement
from dcaf.output.results import assemble_component_output, assemble_output


class TestWeightDomainE2E:
    """Full weight domain pipeline on real tensors."""

    def test_projection_rms_on_real_delta(self, tiny_model_pair):
        base, trained = tiny_model_pair
        base_sd = dict(base.named_parameters())
        trained_sd = dict(trained.named_parameters())

        for name in base_sd:
            if base_sd[name].dim() >= 2:
                delta = trained_sd[name].data - base_sd[name].data
                rms = compute_projection_rms(delta)
                assert rms > 0, f"RMS should be positive for perturbed weights: {name}"
                assert rms < 1.0, f"RMS unreasonably large for small perturbation: {name}"
                break

    def test_significance_pipeline(self, tiny_model_pair):
        base, trained = tiny_model_pair
        rms_values = {}
        for name, p_base in base.named_parameters():
            if p_base.dim() >= 2:
                p_trained = dict(trained.named_parameters())[name]
                delta = p_trained.data - p_base.data
                rms_values[name] = compute_projection_rms(delta)

        assert len(rms_values) > 0
        sig_count = sum(1 for name in rms_values if sig(name, rms_values))
        sigbar_count = sum(1 for name in rms_values if sig_bar(name, rms_values))
        assert sig_count > 0, "At least some projections should be significant"
        assert sigbar_count > 0, "At least some projections should be insignificant"
        assert sig_count + sigbar_count <= len(rms_values)

    def test_opposition_on_synthetic_clusters(self):
        torch.manual_seed(42)
        delta_plus = torch.randn(64, 32) * 0.5
        delta_minus = -delta_plus + torch.randn(64, 32) * 0.1
        cos_opp, opp_deg = compute_opposition_degree(delta_plus, delta_minus)
        assert opp_deg > 0.5, "Opposing deltas should have high opposition"

    def test_confidence_produces_valid_score(self):
        torch.manual_seed(42)
        rms_by_signal = {
            "t1": {"proj_A": 0.5, "proj_B": 0.1, "proj_C": 0.8},
            "t2": {"proj_A": 0.6, "proj_B": 0.05, "proj_C": 0.7},
            "t6": {"proj_A": 0.4, "proj_B": 0.02, "proj_C": 0.6},
        }
        eff = {"t1": 0.9, "t2": 0.8, "t6": 0.7}
        C_W = compute_projection_confidence(
            proj="proj_C",
            rms_by_signal=rms_by_signal,
            effectiveness=eff,
            opp_degree=0.6,
            behavioral_signals=["t1", "t2", "t6"],
            baseline_signals=[],
        )
        assert 0.0 <= C_W <= 1.0

    def test_component_aggregation_via_max(self):
        proj_confidences = {"L5H3_Q": 0.3, "L5H3_K": 0.1, "L5H3_V": 0.9, "L5H3_O": 0.4}
        C_W_component = aggregate_component_confidence(
            component_projs=list(proj_confidences.keys()),
            proj_confidences=proj_confidences,
        )
        assert C_W_component == 0.9, "Component confidence should be MAX of projection confidences"


class TestTriangulationE2E:
    """Full confidence triangulation pipeline."""

    def test_triangulation_with_all_domains(self):
        tri = TriangulatedConfidence.compute(C_W=0.8, C_A=0.6, C_G=0.7)
        assert 0.0 <= tri.value <= 1.0
        assert tri.value > 0.5, "High domain scores should produce high triangulated confidence"

    def test_unified_with_multi_path_bonus(self):
        uc = UnifiedConfidence.compute(C_W=0.8, C_A=0.6, C_G=0.7, path_count=3)
        uc_single = UnifiedConfidence.compute(C_W=0.8, C_A=0.6, C_G=0.7, path_count=1)
        assert uc.value > uc_single.value, "Multi-path should increase confidence"
        assert uc.bonus == 0.30, "3-path bonus = 2 * 0.15 = 0.30"

    def test_low_scores_produce_low_confidence(self):
        tri = TriangulatedConfidence.compute(C_W=0.05, C_A=0.05, C_G=0.05)
        assert tri.value < 0.2


class TestCircuitGraphE2E:
    """Circuit graph construction and serialization."""

    def test_build_small_circuit(self):
        g = CircuitGraph()
        g.add_node("L5H3")
        g.add_node("L10_MLP")
        g.add_edge("L5H3", "L10_MLP", weight=0.85, edge_type="ablation")

        assert len(g.nodes) == 2
        assert len(g.edges) == 1

        d = g.to_dict()
        g2 = CircuitGraph.from_dict(d)
        assert len(g2.nodes) == 2
        assert len(g2.edges) == 1


class TestFullPipelineE2E:
    """End-to-end: weight analysis -> confidence -> classification -> output."""

    def test_full_vertical_slice(self):
        C_W, C_A, C_G = 0.82, 0.65, 0.71

        tri = TriangulatedConfidence.compute(C_W=C_W, C_A=C_A, C_G=C_G)
        uc = UnifiedConfidence.compute(C_W=C_W, C_A=C_A, C_G=C_G, path_count=2)
        assert tri.value > 0

        final = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1,
            component_confidence=uc.value,
            solo_confirmed=True,
        )
        assert final.status == ComponentStatus.CONFIRMED
        assert final.interaction_requirement == InteractionRequirement.SOLO

        comp_out = assemble_component_output(
            component="L5H3", C_W=C_W, C_A=C_A, C_G=C_G,
            unified_confidence=uc.value,
            paths=["W", "A"],
            classification=final.to_dict(),
        )
        output = assemble_output({"L5H3": comp_out})
        assert output["summary"]["total_components"] == 1
        assert output["version"] == "0.1.0"

    def test_pathway_attribution_in_circuit(self):
        torch.manual_seed(42)
        dQ = torch.randn(64, 32) * 0.1
        dK = torch.randn(64, 32) * 0.1
        dV = torch.randn(64, 32) * 2.0

        attr = compute_pathway_from_weight_deltas(dQ, dK, dV, "L5_MLP", "L8H3")
        assert attr.dominant_pathway == "V"

        g = CircuitGraph()
        g.add_node("L5_MLP")
        g.add_node("L8H3")
        g.add_edge("L5_MLP", "L8H3", weight=0.85, edge_type="ablation")

        d = g.to_dict()
        assert len(d["edges"]) == 1
