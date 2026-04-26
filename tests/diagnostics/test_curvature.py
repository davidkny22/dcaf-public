"""
Tests for dcaf.diagnostics.curvature — online curvature tracking.

Covers CurvatureMetrics and OnlineCurvatureTracker dataclasses (creation,
serialization, round-trip), plus the functional API (init, update, finalize)
under straight-path, right-angle, zigzag, zero-movement, and circular
scenarios.
"""

import math
import pytest
import torch
import torch.nn as nn

from dcaf.diagnostics.curvature import (
    CurvatureMetrics,
    OnlineCurvatureTracker,
    init_curvature_tracker,
    update_curvature_tracker,
    finalize_curvature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """A trivial model — the curvature API only passes it through to capture_fn."""
    return nn.Linear(4, 4)


def _capture_fn_from_sequence(tensors):
    """
    Return a capture_fn that yields tensors from a pre-built list in order.

    Each call to the returned function pops the next tensor.  The model,
    component, and probe_data arguments are ignored.
    """
    it = iter(tensors)

    def capture_fn(model, component, probe_data):
        return next(it)

    return capture_fn


# ---------------------------------------------------------------------------
# TestOnlineCurvatureTracker — functional / integration tests
# ---------------------------------------------------------------------------

class TestOnlineCurvatureTracker:

    def test_init_tracker(self):
        """init_curvature_tracker should capture initial activations and set path length to 0."""
        start = torch.tensor([1.0, 2.0, 3.0])
        capture_fn = _capture_fn_from_sequence([start])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L5H3", None, capture_fn)

        assert tracker.component == "L5H3"
        assert torch.allclose(tracker.initial_activations, start)
        assert torch.allclose(tracker.prev_activations, start)
        assert tracker.cumulative_path_length == 0.0

    def test_update_adds_segment_length(self):
        """update_curvature_tracker should add the Euclidean segment length."""
        start = torch.tensor([0.0, 0.0])
        mid = torch.tensor([3.0, 4.0])  # distance = 5.0

        capture_fn = _capture_fn_from_sequence([start, mid])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L1H0", None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)

        assert tracker.cumulative_path_length == pytest.approx(5.0)
        assert torch.allclose(tracker.prev_activations, mid)

    def test_straight_path_curvature_zero(self):
        """
        A -> B in one straight step should yield curvature = 0.

        path_length == direct_distance  =>  curvature = (d/d) - 1 = 0
        """
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([6.0, 8.0])  # distance = 10.0

        # init consumes start; finalize consumes end
        capture_fn = _capture_fn_from_sequence([start, end])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L0H0", None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        assert curvature == pytest.approx(0.0)

    def test_right_angle_path(self):
        """
        A right-angle path is longer than a straight line, so curvature > 0.

        Path: (0,0) -> (3,0) -> (3,4)
        Segment lengths: 3 + 4 = 7
        Direct distance: sqrt(9 + 16) = 5
        Curvature: (7 / 5) - 1 = 0.4
        """
        start = torch.tensor([0.0, 0.0])
        mid = torch.tensor([3.0, 0.0])
        end = torch.tensor([3.0, 4.0])

        capture_fn = _capture_fn_from_sequence([start, mid, end])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L2H1", None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        assert curvature == pytest.approx(0.4)

    def test_zigzag_path_high_curvature(self):
        """
        A zigzag path accumulates more distance than a straight path.

        Path: (0,0) -> (10,0) -> (0,0) -> (10,0)
        Segment lengths: 10 + 10 + 10 = 30
        Direct distance: 10
        Curvature: (30 / 10) - 1 = 2.0
        """
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([10.0, 0.0])
        p2 = torch.tensor([0.0, 0.0])
        p3 = torch.tensor([10.0, 0.0])

        capture_fn = _capture_fn_from_sequence([p0, p1, p2, p3])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L7_MLP", None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        assert curvature == pytest.approx(2.0)

    def test_zero_movement_returns_zero(self):
        """
        No movement at all: direct distance ~ 0 AND path length ~ 0 => curvature = 0.
        """
        same = torch.tensor([1.0, 2.0, 3.0])

        capture_fn = _capture_fn_from_sequence([same, same])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L0H0", None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        assert curvature == 0.0

    def test_circular_return_to_start(self):
        """
        Return to start after moving: direct_distance ~ 0 but path_length > 0 => inf.

        Path: (0,0) -> (5,0) -> (0,0)
        Cumulative path: 5 + 5 = 10
        Direct distance: 0
        """
        origin = torch.tensor([0.0, 0.0])
        away = torch.tensor([5.0, 0.0])
        back = torch.tensor([0.0, 0.0])

        capture_fn = _capture_fn_from_sequence([origin, away, back])
        model = _make_model()

        tracker = init_curvature_tracker(model, "L3H2", None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        assert curvature == float("inf")

    def test_curvature_formula_manual(self):
        """
        Manually verify the curvature formula with specific numbers.

        Path: (0) -> (2) -> (5) -> (11)
        Segments: |2-0| + |5-2| + |11-5| = 2 + 3 + 6 = 11
        Direct: |11-0| = 11
        Curvature: (11 / 11) - 1 = 0.0  (monotonically increasing => straight)
        """
        pts = [
            torch.tensor([0.0]),
            torch.tensor([2.0]),
            torch.tensor([5.0]),
            torch.tensor([11.0]),
        ]

        capture_fn = _capture_fn_from_sequence(pts)
        model = _make_model()

        tracker = init_curvature_tracker(model, "L0_MLP", None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        update_curvature_tracker(tracker, model, None, capture_fn)
        curvature = finalize_curvature(tracker, model, None, capture_fn)

        # All movement in the same direction => perfectly straight
        assert curvature == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestCurvatureMetricsSerialization
# ---------------------------------------------------------------------------

class TestCurvatureMetricsSerialization:

    def test_to_dict(self):
        m = CurvatureMetrics(enabled=True, curvature_plus=0.12, curvature_minus=0.34)
        d = m.to_dict()
        assert d == {
            "enabled": True,
            "curvature_plus": 0.12,
            "curvature_minus": 0.34,
        }

    def test_from_dict(self):
        d = {"enabled": True, "curvature_plus": 0.5, "curvature_minus": 0.9}
        m = CurvatureMetrics.from_dict(d)
        assert m.enabled is True
        assert m.curvature_plus == 0.5
        assert m.curvature_minus == 0.9

    def test_round_trip(self):
        original = CurvatureMetrics(enabled=True, curvature_plus=1.23, curvature_minus=4.56)
        restored = CurvatureMetrics.from_dict(original.to_dict())
        assert restored.enabled == original.enabled
        assert restored.curvature_plus == original.curvature_plus
        assert restored.curvature_minus == original.curvature_minus

    def test_disabled(self):
        m = CurvatureMetrics(enabled=False)
        d = m.to_dict()
        assert d["enabled"] is False
        assert d["curvature_plus"] is None
        assert d["curvature_minus"] is None

        restored = CurvatureMetrics.from_dict(d)
        assert restored.enabled is False
        assert restored.curvature_plus is None
        assert restored.curvature_minus is None


# ---------------------------------------------------------------------------
# TestOnlineCurvatureTrackerSerialization
# ---------------------------------------------------------------------------

class TestOnlineCurvatureTrackerSerialization:

    def test_to_dict(self):
        tracker = OnlineCurvatureTracker(
            component="L5H3",
            initial_activations=torch.tensor([1.0, 2.0]),
            prev_activations=torch.tensor([3.0, 4.0]),
            cumulative_path_length=7.5,
        )
        d = tracker.to_dict()
        assert d["component"] == "L5H3"
        assert d["initial_activations"] == [1.0, 2.0]
        assert d["prev_activations"] == [3.0, 4.0]
        assert d["cumulative_path_length"] == 7.5

    def test_from_dict(self):
        d = {
            "component": "L12_MLP",
            "initial_activations": [0.0, 1.0, 2.0],
            "prev_activations": [3.0, 4.0, 5.0],
            "cumulative_path_length": 12.0,
        }
        tracker = OnlineCurvatureTracker.from_dict(d)
        assert tracker.component == "L12_MLP"
        assert torch.allclose(tracker.initial_activations, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.allclose(tracker.prev_activations, torch.tensor([3.0, 4.0, 5.0]))
        assert tracker.cumulative_path_length == 12.0

    def test_round_trip(self):
        original = OnlineCurvatureTracker(
            component="L8H0",
            initial_activations=torch.tensor([10.0, 20.0, 30.0]),
            prev_activations=torch.tensor([11.0, 22.0, 33.0]),
            cumulative_path_length=99.9,
        )
        restored = OnlineCurvatureTracker.from_dict(original.to_dict())

        assert restored.component == original.component
        assert torch.allclose(restored.initial_activations, original.initial_activations)
        assert torch.allclose(restored.prev_activations, original.prev_activations)
        assert restored.cumulative_path_length == pytest.approx(original.cumulative_path_length)
