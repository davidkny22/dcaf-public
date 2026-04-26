"""Test signal definitions (Def 1.6-1.7)."""

from dcaf.core.signals import (
    CANONICAL_SIGNALS,
    TrainingSignal,
    get_target_signals,
    get_opposite_signals,
    get_baseline_signals,
    get_behavioral_signals,
)


class TestCanonicalSignals:
    def test_total_count(self):
        assert len(CANONICAL_SIGNALS) == 11

    def test_target_cluster_count(self):
        assert len(get_target_signals()) == 5

    def test_opposite_cluster_count(self):
        assert len(get_opposite_signals()) == 5

    def test_baseline_cluster_count(self):
        assert len(get_baseline_signals()) == 1

    def test_behavioral_excludes_baseline(self):
        behavioral = get_behavioral_signals()
        assert len(behavioral) == 10
        assert all(s.cluster != "0" for s in behavioral)

    def test_signal_ids_unique(self):
        ids = [s.id for s in CANONICAL_SIGNALS]
        assert len(ids) == len(set(ids))

    def test_signal_ids_sequential(self):
        for i, s in enumerate(CANONICAL_SIGNALS, 1):
            assert s.id == f"t{i}"

    def test_t1_is_prefopt_target(self):
        t1 = CANONICAL_SIGNALS[0]
        assert t1.cluster == "+"
        assert t1.signal_type == "PrefOpt"

    def test_t6_is_prefopt_opposite(self):
        t6 = CANONICAL_SIGNALS[5]
        assert t6.cluster == "-"
        assert t6.signal_type == "PrefOpt"

    def test_t11_is_domain_native(self):
        t11 = CANONICAL_SIGNALS[10]
        assert t11.cluster == "0"
        assert t11.signal_type == "DomainNative"

    def test_target_opposite_symmetry(self):
        target = get_target_signals()
        opposite = get_opposite_signals()
        target_types = [s.signal_type for s in target]
        opposite_types = [s.signal_type for s in opposite]
        assert target_types == opposite_types


class TestTrainingSignalSerialization:
    def test_round_trip(self):
        s = TrainingSignal("t1", "PrefOpt(target>opposite)", "+", "PrefOpt", 0.85)
        d = s.to_dict()
        s2 = TrainingSignal.from_dict(d)
        assert s2.id == s.id
        assert s2.cluster == s.cluster
        assert s2.effectiveness == s.effectiveness

    def test_default_effectiveness(self):
        d = {"id": "t1", "name": "test", "cluster": "+", "signal_type": "SFT"}
        s = TrainingSignal.from_dict(d)
        assert s.effectiveness == 0.0
