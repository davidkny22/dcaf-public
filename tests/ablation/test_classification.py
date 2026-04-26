"""Test Phase 7 unified classifier (Def 11.20, 11.23-11.27)."""

import pytest

from dcaf.ablation.classification import (
    ComponentStatus,
    FinalClassification,
    classify_all_final,
    classify_final,
)
from dcaf.ablation.superadditivity import InteractionRequirement, InteractionType
from dcaf.circuit.classification import classify_component_tiered


class TestTieredClassification:
    """Test adaptive tiered classification (Def 11.23-11.27)."""

    def test_all_below_tau_abs_is_false_positive(self):
        result = classify_component_tiered("L5H3", I_detect=0.1, I_decide=0.05, I_eval=0.02)
        assert result.status == "FalsePositive"
        assert len(result.primary) == 0

    def test_single_dominant_probe(self):
        result = classify_component_tiered("L5H3", I_detect=0.9, I_decide=0.1, I_eval=0.05)
        assert result.status == "Confirmed"
        assert len(result.primary) == 1
        assert result.primary[0]["function"] == "Recognition"

    def test_tight_cluster_uses_relaxed_threshold(self):
        result = classify_component_tiered("L5H3", I_detect=0.5, I_decide=0.45, I_eval=0.42)
        assert result.tight_cluster is True
        assert result.threshold_used == 0.7

    def test_spread_uses_strict_threshold(self):
        result = classify_component_tiered("L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1)
        assert result.tight_cluster is False
        assert result.threshold_used == 0.8

    def test_diffuse_flag_with_three_primaries(self):
        result = classify_component_tiered("L5H3", I_detect=0.5, I_decide=0.48, I_eval=0.46)
        assert result.tight_cluster is True
        assert result.diffuse is True
        assert len(result.primary) == 3

    def test_auxiliary_tier(self):
        result = classify_component_tiered("L5H3", I_detect=0.9, I_decide=0.6, I_eval=0.05)
        primaries = {p["function"] for p in result.primary}
        auxiliaries = {a["function"] for a in result.auxiliary}
        assert "Recognition" in primaries
        assert "Steering" in auxiliaries or "Steering" in primaries

    def test_orphan_status_with_high_confidence(self):
        result = classify_component_tiered(
            "L5H3", I_detect=0.1, I_decide=0.05, I_eval=0.02, component_confidence=0.8
        )
        assert result.status == "FalsePositive"

    def test_serialization_round_trip(self):
        result = classify_component_tiered("L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1)
        d = result.to_dict()
        from dcaf.circuit.classification import TieredClassification
        restored = TieredClassification.from_dict(d)
        assert restored.component == result.component
        assert restored.status == result.status
        assert len(restored.primary) == len(result.primary)


class TestFinalClassifier:
    """Test Phase 7 unified ORPHAN/SOLO/PAIR/GATE (Def 11.20)."""

    def test_solo_confirmed(self):
        result = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1, solo_confirmed=True
        )
        assert result.interaction_requirement == InteractionRequirement.SOLO
        assert result.status == ComponentStatus.CONFIRMED

    def test_pair_confirmed(self):
        result = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1,
            pair_partner="L10_MLP", pair_interaction_type=InteractionType.SYNERGISTIC,
        )
        assert result.interaction_requirement == InteractionRequirement.PAIR
        assert result.interaction_partner == "L10_MLP"
        assert result.interaction_type == InteractionType.SYNERGISTIC

    def test_gate_confirmed(self):
        result = classify_final(
            "L5H3", I_detect=0.7, I_decide=0.5, I_eval=0.3, in_gate_triple=True
        )
        assert result.interaction_requirement == InteractionRequirement.GATE

    def test_orphan_when_nothing_confirmed(self):
        result = classify_final(
            "L5H3", I_detect=0.5, I_decide=0.3, I_eval=0.2,
            component_confidence=0.7,
        )
        assert result.status == ComponentStatus.ORPHAN

    def test_false_positive_when_low_impact(self):
        result = classify_final("L5H3", I_detect=0.05, I_decide=0.02, I_eval=0.01)
        assert result.status == ComponentStatus.FALSE_POSITIVE

    def test_promoted_orphan_gets_pair(self):
        result = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1,
            is_orphan_retested=True, orphan_confirmed=True,
        )
        assert result.interaction_requirement == InteractionRequirement.PAIR

    def test_gate_takes_priority_over_pair(self):
        result = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1,
            pair_partner="L10_MLP", in_gate_triple=True,
        )
        assert result.interaction_requirement == InteractionRequirement.GATE

    def test_to_dict(self):
        result = classify_final(
            "L5H3", I_detect=0.9, I_decide=0.3, I_eval=0.1,
            solo_confirmed=True, discovery_count=2,
            strategies_found_by=["A_graph_adjacent", "E_opposition"],
        )
        d = result.to_dict()
        assert d["component"] == "L5H3"
        assert d["interaction_requirement"] == "solo"
        assert d["discovery_count"] == 2
        assert len(d["strategies_found_by"]) == 2


class TestClassifyAllFinal:
    def test_batch_classification(self):
        components = {
            "L5H3": {"I_detect": 0.9, "I_decide": 0.1, "I_eval": 0.05},
            "L10_MLP": {"I_detect": 0.3, "I_decide": 0.8, "I_eval": 0.2},
            "L12H1": {"I_detect": 0.02, "I_decide": 0.01, "I_eval": 0.01},
        }
        results = classify_all_final(
            components=components,
            solo_set={"L5H3", "L10_MLP"},
            pair_results={"L5H3": ("L10_MLP", InteractionType.SYNERGISTIC)},
            gate_components=set(),
            orphan_confirmed=set(),
        )
        assert results["L5H3"].interaction_requirement == InteractionRequirement.PAIR
        assert results["L10_MLP"].interaction_requirement == InteractionRequirement.SOLO
        assert results["L12H1"].status == ComponentStatus.FALSE_POSITIVE
