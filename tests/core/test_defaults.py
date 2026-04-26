"""Verify all SS20.13 threshold constants have correct values."""

from dcaf.core.defaults import (
    TAU_SIG, TAU_BASE, TAU_ACT, TAU_GRAD, TAU_COMP,
    ALPHA, TAU_OPP, BETA, Q, BETA_PATH,
    P_LRS, EPSILON_TRI, LRS_NONLINEAR_THRESHOLD,
    PEAK_CONFIRMATION_WINDOW, PEAK_STABILITY_TOLERANCE,
    R_STRICT_PRIMARY, R_RELAXED_PRIMARY, R_AUX, TAU_ABS, TAU_GAP,
    PAIR_BUDGET, TRIPLE_BUDGET, SYNERGY_EPSILON,
    EPS_RMS, W_DISCOVERY,
)


class TestDiscoveryThresholds:
    def test_tau_sig(self):
        assert TAU_SIG == 85.0

    def test_tau_base(self):
        assert TAU_BASE == 50.0

    def test_tau_act(self):
        assert TAU_ACT == 85.0

    def test_tau_grad(self):
        assert TAU_GRAD == 85.0

    def test_tau_comp(self):
        assert TAU_COMP == 70.0


class TestWeightDomainParams:
    def test_alpha(self):
        assert ALPHA == 0.2

    def test_tau_opp(self):
        assert TAU_OPP == 0.3

    def test_beta(self):
        assert BETA == 0.4

    def test_q(self):
        assert Q == 2

    def test_beta_path(self):
        assert BETA_PATH == 0.15


class TestGeometryParams:
    def test_p_lrs(self):
        assert P_LRS == 0.5

    def test_epsilon_tri(self):
        assert EPSILON_TRI == 0.05

    def test_lrs_nonlinear_threshold(self):
        assert LRS_NONLINEAR_THRESHOLD == 0.4


class TestPeakCheckpointParams:
    def test_confirmation_window(self):
        assert PEAK_CONFIRMATION_WINDOW == 3

    def test_stability_tolerance(self):
        assert PEAK_STABILITY_TOLERANCE == 0.05


class TestClassificationParams:
    def test_r_strict_primary(self):
        assert R_STRICT_PRIMARY == 0.8

    def test_r_relaxed_primary(self):
        assert R_RELAXED_PRIMARY == 0.7

    def test_r_aux(self):
        assert R_AUX == 0.6

    def test_tau_abs(self):
        assert TAU_ABS == 0.15

    def test_tau_gap(self):
        assert TAU_GAP == 0.10


class TestAblationParams:
    def test_pair_budget(self):
        assert PAIR_BUDGET == 300

    def test_triple_budget(self):
        assert TRIPLE_BUDGET == 100

    def test_synergy_epsilon(self):
        assert SYNERGY_EPSILON == 0.05


class TestNumericalParams:
    def test_eps_rms(self):
        assert EPS_RMS == 1e-8

    def test_w_discovery(self):
        assert W_DISCOVERY == 2
