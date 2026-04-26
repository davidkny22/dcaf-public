"""
Tests for dcaf.circuit.component_result — ComponentResult serialization.

Covers basic creation, full-key round-trip serialization, diagnostic
sub-object attachment (ActivationDeltaAlignment, CurvatureMetrics),
and edge cases (empty lists, None unified score, repr).
"""

import pytest

from dcaf.circuit.component_result import ComponentResult
from dcaf.diagnostics.alignment import ActivationDeltaAlignment
from dcaf.diagnostics.curvature import CurvatureMetrics


# ---------------------------------------------------------------------------
# TestComponentResultBasic
# ---------------------------------------------------------------------------

class TestComponentResultBasic:

    def test_minimal_creation(self):
        """Creating a ComponentResult with only the component name should work."""
        cr = ComponentResult(component="L10_MLP")
        assert cr.component == "L10_MLP"
        assert cr.param_names == []
        assert cr.C_W is None
        assert cr.C_A is None
        assert cr.C_G is None
        assert cr.C_unified is None
        assert cr.ablation_status == "untested"

    def test_to_dict_has_all_keys(self):
        """to_dict should include every expected top-level key."""
        cr = ComponentResult(component="L5H3")
        d = cr.to_dict()

        expected_keys = {
            "component",
            "param_names",
            "scores",
            "discovery",
            "weight_details",
            "geometry_details",
            "classification",
            "interaction",
            "ablation",
            "diagnostics",
            "delta_alignment",
            "curvature",
            "nonlinear",
            "probing",
        }
        assert set(d.keys()) == expected_keys

        # Nested score keys
        assert set(d["scores"].keys()) == {"C_W", "C_A", "C_G", "C_unified"}
        # Nested discovery keys
        assert set(d["discovery"].keys()) == {"paths", "path_count", "multi_path_bonus"}
        # Nested interaction keys
        assert set(d["interaction"].keys()) == {"requirement", "partners", "type"}
        # Nested ablation keys
        assert set(d["ablation"].keys()) == {"confirmed", "status"}

    def test_from_dict_round_trip(self):
        """A fully populated ComponentResult should survive to_dict -> from_dict."""
        original = ComponentResult(
            component="L7H2",
            param_names=["transformer.h.7.attn.c_attn.weight"],
            C_W=0.85,
            C_A=0.72,
            C_G=0.65,
            C_unified=0.80,
            discovery_paths={"W", "A"},
            path_count=2,
            multi_path_bonus=0.05,
            bidirectional=True,
            opp_degree=0.91,
            lrs=0.88,
            lrs_breakdown={"coh_plus": 0.9, "coh_minus": 0.85},
            classification={"tier": "primary", "label": "causal"},
            interaction_requirement="solo",
            interaction_partners=[],
            interaction_type="additive",
            ablation_confirmed=True,
            ablation_status="behavioral",
            diagnostics={"deviation": 0.02},
        )

        restored = ComponentResult.from_dict(original.to_dict())

        assert restored.component == original.component
        assert restored.param_names == original.param_names
        assert restored.C_W == original.C_W
        assert restored.C_A == original.C_A
        assert restored.C_G == original.C_G
        assert restored.C_unified == original.C_unified
        assert restored.discovery_paths == original.discovery_paths
        assert restored.path_count == original.path_count
        assert restored.multi_path_bonus == original.multi_path_bonus
        assert restored.bidirectional == original.bidirectional
        assert restored.opp_degree == original.opp_degree
        assert restored.lrs == original.lrs
        assert restored.lrs_breakdown == original.lrs_breakdown
        assert restored.classification == original.classification
        assert restored.interaction_requirement == original.interaction_requirement
        assert restored.interaction_partners == original.interaction_partners
        assert restored.interaction_type == original.interaction_type
        assert restored.ablation_confirmed == original.ablation_confirmed
        assert restored.ablation_status == original.ablation_status
        assert restored.diagnostics == original.diagnostics


# ---------------------------------------------------------------------------
# TestComponentResultDiagnostics
# ---------------------------------------------------------------------------

class TestComponentResultDiagnostics:

    def test_with_delta_alignment(self):
        """
        Attaching an ActivationDeltaAlignment should serialize and
        deserialize correctly through ComponentResult.
        """
        alignment = ActivationDeltaAlignment(
            align_plus=0.85,
            align_minus=0.78,
            opposition=-0.62,
        )
        cr = ComponentResult(component="L10H3", delta_alignment=alignment)

        d = cr.to_dict()
        assert d["delta_alignment"] is not None
        assert d["delta_alignment"]["align_plus"] == 0.85
        assert d["delta_alignment"]["align_minus"] == 0.78
        assert d["delta_alignment"]["opposition"] == -0.62

        restored = ComponentResult.from_dict(d)
        assert isinstance(restored.delta_alignment, ActivationDeltaAlignment)
        assert restored.delta_alignment.align_plus == 0.85
        assert restored.delta_alignment.align_minus == 0.78
        assert restored.delta_alignment.opposition == -0.62

    def test_with_curvature_metrics(self):
        """
        Attaching CurvatureMetrics should serialize and deserialize correctly.
        """
        curv = CurvatureMetrics(
            enabled=True,
            curvature_plus=0.12,
            curvature_minus=0.34,
        )
        cr = ComponentResult(component="L5_MLP", curvature=curv)

        d = cr.to_dict()
        assert d["curvature"] is not None
        assert d["curvature"]["enabled"] is True
        assert d["curvature"]["curvature_plus"] == 0.12
        assert d["curvature"]["curvature_minus"] == 0.34

        restored = ComponentResult.from_dict(d)
        assert isinstance(restored.curvature, CurvatureMetrics)
        assert restored.curvature.enabled is True
        assert restored.curvature.curvature_plus == 0.12
        assert restored.curvature.curvature_minus == 0.34

    def test_all_diagnostics_none(self):
        """
        When all optional diagnostic fields are None, serialization should
        still produce valid output with None values.
        """
        cr = ComponentResult(component="L0H0")
        d = cr.to_dict()

        assert d["delta_alignment"] is None
        assert d["curvature"] is None
        assert d["nonlinear"] is None
        assert d["probing"] is None

        restored = ComponentResult.from_dict(d)
        assert restored.delta_alignment is None
        assert restored.curvature is None
        assert restored.nonlinear is None
        assert restored.probing is None

    def test_partial_diagnostics(self):
        """
        When some diagnostic fields are set and others are None, the round
        trip should preserve both states correctly.
        """
        alignment = ActivationDeltaAlignment(
            align_plus=0.90,
            align_minus=0.80,
            opposition=-0.70,
        )
        cr = ComponentResult(
            component="L3H1",
            delta_alignment=alignment,
            curvature=None,
            nonlinear=None,
            probing=None,
        )

        d = cr.to_dict()
        assert d["delta_alignment"] is not None
        assert d["curvature"] is None
        assert d["nonlinear"] is None
        assert d["probing"] is None

        restored = ComponentResult.from_dict(d)
        assert isinstance(restored.delta_alignment, ActivationDeltaAlignment)
        assert restored.delta_alignment.align_plus == 0.90
        assert restored.curvature is None
        assert restored.nonlinear is None
        assert restored.probing is None


# ---------------------------------------------------------------------------
# TestComponentResultEdgeCases
# ---------------------------------------------------------------------------

class TestComponentResultEdgeCases:

    def test_empty_param_names(self):
        """An empty param_names list should round-trip cleanly."""
        cr = ComponentResult(component="L0_MLP", param_names=[])
        restored = ComponentResult.from_dict(cr.to_dict())
        assert restored.param_names == []

    def test_empty_discovery_paths(self):
        """An empty discovery_paths set should round-trip as an empty set."""
        cr = ComponentResult(component="L1H0", discovery_paths=set())
        d = cr.to_dict()
        assert d["discovery"]["paths"] == []

        restored = ComponentResult.from_dict(d)
        assert restored.discovery_paths == set()

    def test_repr_with_none_unified(self):
        """
        __repr__ formats C_unified as 'None' when it is None.
        Should not raise an error.
        """
        cr = ComponentResult(component="L2H5", C_unified=None)
        text = repr(cr)
        assert "L2H5" in text
        assert "None" in text

        # Also test with an actual value
        cr2 = ComponentResult(component="L2H5", C_unified=0.75)
        text2 = repr(cr2)
        assert "0.750" in text2
