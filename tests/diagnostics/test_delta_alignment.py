"""
Tests for dcaf.diagnostics.alignment module.

Covers:
- _mean_pairwise_cosine: intra-group cosine similarity
- _mean_cross_cosine: cross-group cosine similarity
- compute_activation_delta_alignment: end-to-end delta alignment computation
- ActivationDeltaAlignment: serialization round-trip
"""

from typing import Dict, List

import pytest
import torch

from dcaf.diagnostics.alignment import (
    ActivationDeltaAlignment,
    _mean_pairwise_cosine,
    _mean_cross_cosine,
    compute_activation_delta_alignment,
)
from tests.conftest import MockSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPONENT = "L10H3"


def _make_activation_deltas(
    signal_ids: List[str],
    component: str,
    tensors: List[torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Build an activation_deltas mapping from parallel lists."""
    deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    for sid, t in zip(signal_ids, tensors):
        deltas[sid] = {component: t}
    return deltas


# ===========================================================================
# TestMeanPairwiseCosine
# ===========================================================================


class TestMeanPairwiseCosine:
    """Tests for _mean_pairwise_cosine."""

    def test_identical_vectors_returns_1(self):
        """Two copies of the same vector should have cosine similarity 1.0."""
        torch.manual_seed(42)
        v = torch.randn(16)
        result = _mean_pairwise_cosine([v, v.clone()])
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_returns_0(self):
        """Two orthogonal vectors should have cosine similarity ~0.0."""
        # Construct perfectly orthogonal vectors using standard basis
        v1 = torch.zeros(32)
        v1[0] = 1.0
        v2 = torch.zeros(32)
        v2[1] = 1.0
        result = _mean_pairwise_cosine([v1, v2])
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_returns_neg1(self):
        """A vector and its negation should have cosine similarity -1.0."""
        torch.manual_seed(42)
        v = torch.randn(16)
        result = _mean_pairwise_cosine([v, -v])
        assert result == pytest.approx(-1.0, abs=1e-5)

    def test_single_vector_returns_0(self):
        """With fewer than 2 vectors, the function returns 0.0."""
        torch.manual_seed(42)
        v = torch.randn(16)
        assert _mean_pairwise_cosine([v]) == 0.0

    def test_empty_list_returns_0(self):
        """An empty list returns 0.0."""
        assert _mean_pairwise_cosine([]) == 0.0

    def test_mixed_similarity(self):
        """Three vectors with known pairwise relationships.

        v1 and v2 are identical (cos=1), v1 and v3 are opposite (cos=-1),
        v2 and v3 are opposite (cos=-1). Mean = (1 + -1 + -1) / 3 = -1/3.
        """
        torch.manual_seed(42)
        v1 = torch.randn(16)
        v2 = v1.clone()
        v3 = -v1
        result = _mean_pairwise_cosine([v1, v2, v3])
        expected = (1.0 + (-1.0) + (-1.0)) / 3.0
        assert result == pytest.approx(expected, abs=1e-5)


# ===========================================================================
# TestMeanCrossCosine
# ===========================================================================


class TestMeanCrossCosine:
    """Tests for _mean_cross_cosine."""

    def test_same_direction_returns_positive(self):
        """Vectors pointing in the same direction across groups give positive cosine."""
        torch.manual_seed(42)
        v = torch.randn(16)
        # Group A and B both contain the same vector
        result = _mean_cross_cosine([v], [v.clone()])
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_opposing_directions_returns_negative(self):
        """Opposite vectors across groups give negative cosine."""
        torch.manual_seed(42)
        v = torch.randn(16)
        result = _mean_cross_cosine([v], [-v])
        assert result == pytest.approx(-1.0, abs=1e-5)

    def test_empty_group_returns_0(self):
        """If either group is empty, result is 0.0."""
        torch.manual_seed(42)
        v = torch.randn(16)
        assert _mean_cross_cosine([], [v]) == 0.0
        assert _mean_cross_cosine([v], []) == 0.0
        assert _mean_cross_cosine([], []) == 0.0

    def test_single_pair(self):
        """One vector in each group: result equals their cosine similarity."""
        # Use orthogonal vectors
        v1 = torch.zeros(32)
        v1[0] = 1.0
        v2 = torch.zeros(32)
        v2[1] = 1.0
        result = _mean_cross_cosine([v1], [v2])
        assert result == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# TestComputeActivationDeltaAlignment
# ===========================================================================


class TestComputeActivationDeltaAlignment:
    """Tests for compute_activation_delta_alignment."""

    def test_high_alignment_same_cluster(self):
        """T+ signals with similar deltas should produce high align_plus."""
        torch.manual_seed(42)
        # Two T+ deltas pointing in similar directions (positive bias)
        d1 = torch.randn(16) * 0.1 + 1.0  # mostly positive
        d2 = torch.randn(16) * 0.1 + 1.0  # mostly positive
        # One T- delta for a complete result
        d3 = torch.randn(16) * 0.1 - 1.0  # mostly negative

        signals = [
            MockSignal(id="s_plus_1", cluster="+"),
            MockSignal(id="s_plus_2", cluster="+"),
            MockSignal(id="s_minus_1", cluster="-"),
        ]
        deltas = _make_activation_deltas(
            ["s_plus_1", "s_plus_2", "s_minus_1"],
            COMPONENT,
            [d1, d2, d3],
        )

        result = compute_activation_delta_alignment(COMPONENT, signals, deltas)
        assert result is not None
        # Two very similar T+ vectors should be highly aligned
        assert result.align_plus > 0.8

    def test_strong_opposition_cross_cluster(self):
        """T+ and T- deltas pointing in opposite directions give negative opposition."""
        torch.manual_seed(42)
        base = torch.randn(32)
        d_plus_1 = base + torch.randn(32) * 0.05
        d_plus_2 = base + torch.randn(32) * 0.05
        d_minus_1 = -base + torch.randn(32) * 0.05
        d_minus_2 = -base + torch.randn(32) * 0.05

        signals = [
            MockSignal(id="p1", cluster="+"),
            MockSignal(id="p2", cluster="+"),
            MockSignal(id="m1", cluster="-"),
            MockSignal(id="m2", cluster="-"),
        ]
        deltas = _make_activation_deltas(
            ["p1", "p2", "m1", "m2"],
            COMPONENT,
            [d_plus_1, d_plus_2, d_minus_1, d_minus_2],
        )

        result = compute_activation_delta_alignment(COMPONENT, signals, deltas)
        assert result is not None
        # Intra-cluster alignment should be high
        assert result.align_plus > 0.9
        assert result.align_minus > 0.9
        # Cross-cluster opposition should be strongly negative
        assert result.opposition < -0.9

    def test_no_deltas_returns_none(self):
        """When no signals have matching deltas, return None."""
        signals = [
            MockSignal(id="p1", cluster="+"),
            MockSignal(id="m1", cluster="-"),
        ]
        # Empty activation_deltas dict
        result = compute_activation_delta_alignment(COMPONENT, signals, {})
        assert result is None

    def test_only_plus_signals(self):
        """With only T+ signals, align_minus and opposition should be 0.0."""
        torch.manual_seed(42)
        d1 = torch.randn(16)
        d2 = d1 + torch.randn(16) * 0.05  # similar direction

        signals = [
            MockSignal(id="p1", cluster="+"),
            MockSignal(id="p2", cluster="+"),
        ]
        deltas = _make_activation_deltas(["p1", "p2"], COMPONENT, [d1, d2])

        result = compute_activation_delta_alignment(COMPONENT, signals, deltas)
        assert result is not None
        assert result.align_plus > 0.5  # some alignment expected
        assert result.align_minus == 0.0  # no minus signals
        assert result.opposition == 0.0  # no cross-cluster pairs

    def test_only_minus_signals(self):
        """With only T- signals, align_plus and opposition should be 0.0."""
        torch.manual_seed(42)
        d1 = torch.randn(16)
        d2 = d1 + torch.randn(16) * 0.05

        signals = [
            MockSignal(id="m1", cluster="-"),
            MockSignal(id="m2", cluster="-"),
        ]
        deltas = _make_activation_deltas(["m1", "m2"], COMPONENT, [d1, d2])

        result = compute_activation_delta_alignment(COMPONENT, signals, deltas)
        assert result is not None
        assert result.align_plus == 0.0  # no plus signals
        assert result.align_minus > 0.5  # some alignment expected
        assert result.opposition == 0.0  # no cross-cluster pairs

    def test_skips_T0_signals(self, mock_signals_plus_minus_zero):
        """T0 signals should be ignored entirely, even if deltas exist."""
        torch.manual_seed(42)
        # Provide deltas only for the T0 signal
        deltas = {
            "t11_language": {COMPONENT: torch.randn(16)},
        }
        result = compute_activation_delta_alignment(
            COMPONENT, mock_signals_plus_minus_zero, deltas,
        )
        # T0 is skipped; no T+/T- deltas found
        assert result is None

    def test_empty_tensors_skipped(self):
        """Tensors with numel()==0 are skipped (not included in computation)."""
        torch.manual_seed(42)
        signals = [
            MockSignal(id="p1", cluster="+"),
            MockSignal(id="p2", cluster="+"),
        ]
        deltas = {
            "p1": {COMPONENT: torch.tensor([])},   # empty tensor
            "p2": {COMPONENT: torch.tensor([])},   # empty tensor
        }
        result = compute_activation_delta_alignment(COMPONENT, signals, deltas)
        # Both deltas are empty, so no valid deltas remain
        assert result is None

    def test_uses_fixture_signals(self, mock_signals_plus_minus_zero):
        """Integration test using the shared fixture with all three clusters."""
        torch.manual_seed(42)
        base = torch.randn(24)
        deltas = {
            "t1_pref_target":  {COMPONENT: base + torch.randn(24) * 0.02},
            "t2_sft_target":   {COMPONENT: base + torch.randn(24) * 0.02},
            "t6_pref_opposite": {COMPONENT: -base + torch.randn(24) * 0.02},
            "t7_sft_opposite":  {COMPONENT: -base + torch.randn(24) * 0.02},
            "t11_language":     {COMPONENT: torch.randn(24) * 0.5},
        }

        result = compute_activation_delta_alignment(
            COMPONENT, mock_signals_plus_minus_zero, deltas,
        )
        assert result is not None
        # T+ signals are nearly identical
        assert result.align_plus > 0.9
        # T- signals are nearly identical
        assert result.align_minus > 0.9
        # T+ vs T- are opposing
        assert result.opposition < -0.9


# ===========================================================================
# TestActivationDeltaAlignmentSerialization
# ===========================================================================


class TestActivationDeltaAlignmentSerialization:
    """Tests for ActivationDeltaAlignment to_dict / from_dict."""

    def test_to_dict(self):
        """to_dict produces expected keys and values."""
        obj = ActivationDeltaAlignment(
            align_plus=0.85,
            align_minus=0.72,
            opposition=-0.63,
        )
        d = obj.to_dict()
        assert d == {
            "align_plus": 0.85,
            "align_minus": 0.72,
            "opposition": -0.63,
        }

    def test_from_dict(self):
        """from_dict reconstructs the dataclass correctly."""
        data = {
            "align_plus": 0.91,
            "align_minus": 0.88,
            "opposition": -0.55,
        }
        obj = ActivationDeltaAlignment.from_dict(data)
        assert obj.align_plus == 0.91
        assert obj.align_minus == 0.88
        assert obj.opposition == -0.55

    def test_round_trip(self):
        """Serializing then deserializing produces an equivalent object."""
        original = ActivationDeltaAlignment(
            align_plus=0.77,
            align_minus=0.65,
            opposition=-0.42,
        )
        restored = ActivationDeltaAlignment.from_dict(original.to_dict())
        assert restored.align_plus == original.align_plus
        assert restored.align_minus == original.align_minus
        assert restored.opposition == original.opposition

    def test_from_dict_defaults(self):
        """from_dict uses 0.0 for missing keys."""
        obj = ActivationDeltaAlignment.from_dict({})
        assert obj.align_plus == 0.0
        assert obj.align_minus == 0.0
        assert obj.opposition == 0.0
