"""DCAF Configuration. All defaults from dcaf.core.defaults (single source of truth)."""

from dataclasses import dataclass
from typing import Optional

from dcaf.core.defaults import (
    TAU_SIG, TAU_BASE, ALPHA, TAU_OPP, Q, BETA, BETA_PATH,
    TAU_ACT, TAU_COMP, TAU_GRAD,
    P_LRS, LAMBDA_CONTRASTIVE,
    W_DISCOVERY, EPSILON_TRI,
    TAU_W_DEFAULT, TAU_A_DEFAULT, TAU_G_DEFAULT, TAU_UNIFIED_DEFAULT,
    TAU_EDGE, PAIR_BUDGET, TRIPLE_BUDGET, SYNERGY_EPSILON, EPSILON_OPP,
    R_STRICT_PRIMARY, R_RELAXED_PRIMARY, R_AUX, TAU_ABS, TAU_GAP, TAU_ORPHAN, TAU_PPL,
    PEAK_EVAL_INTERVAL, PEAK_CONFIRMATION_WINDOW, PEAK_STABILITY_TOLERANCE,
    LRS_NONLINEAR_THRESHOLD, EPS_RMS,
    NUM_TRAIN_EPOCHS, MAX_TRAIN_STEPS,
    LEARNING_RATE_SFT, SIMPO_LEARNING_RATE, WARMUP_RATIO, MAX_GRAD_NORM,
    SIMPO_BETA, SIMPO_BATCH_SIZE, SIMPO_GRAD_ACCUM,
    SFT_BATCH_SIZE, SFT_GRAD_ACCUM,
)


@dataclass
class DCAFConfig:
    """Configuration for Differential Circuit Analysis Framework.

    Organized by spec labels:
      def:threshold-parameters: discovery thresholds
      sec:weight-analysis: weight domain parameters
      sec:geometry-analysis: geometry domain parameters
      sec:unified-confidence: confidence and candidate selection
      sec:circuit-graph: circuit graph
      sec:ablation: ablation budgets and interaction thresholds
      sec:adaptive-tiered-functional-classification: classification thresholds
    """

    # === def:threshold-parameters: DISCOVERY THRESHOLDS ===
    tau_sig: float = TAU_SIG
    tau_base: float = TAU_BASE
    tau_act: float = TAU_ACT
    tau_grad: float = TAU_GRAD
    tau_comp: float = TAU_COMP

    # === sec:weight-analysis: WEIGHT DOMAIN PARAMETERS ===
    alpha: float = ALPHA
    tau_opp: float = TAU_OPP
    beta: float = BETA
    q: int = Q

    # === sec:geometry-analysis: GEOMETRY PARAMETERS ===
    p: float = P_LRS
    epsilon: float = EPSILON_TRI
    lambda_reg: float = LAMBDA_CONTRASTIVE

    # === sec:unified-confidence: CONFIDENCE & CANDIDATE SELECTION ===
    w: int = W_DISCOVERY
    beta_path: float = BETA_PATH
    tau_W: float = TAU_W_DEFAULT
    tau_A: float = TAU_A_DEFAULT
    tau_G: float = TAU_G_DEFAULT
    tau_unified: float = TAU_UNIFIED_DEFAULT

    # === sec:circuit-graph: CIRCUIT GRAPH ===
    tau_E: float = TAU_EDGE

    # === sec:ablation: ABLATION ===
    pair_budget: int = PAIR_BUDGET
    triple_budget: int = TRIPLE_BUDGET
    synergy_epsilon: float = SYNERGY_EPSILON
    epsilon_opp: float = EPSILON_OPP
    tau_orphan: float = TAU_ORPHAN

    # === sec:adaptive-tiered-functional-classification: ADAPTIVE TIERED CLASSIFICATION ===
    r_strict_primary: float = R_STRICT_PRIMARY
    r_relaxed_primary: float = R_RELAXED_PRIMARY
    r_aux: float = R_AUX
    tau_abs: float = TAU_ABS
    tau_gap: float = TAU_GAP
    tau_ppl: float = TAU_PPL

    # === PEAK CHECKPOINT (def:peak-checkpoint) ===
    peak_eval_interval: int = PEAK_EVAL_INTERVAL
    peak_confirmation_window: int = PEAK_CONFIRMATION_WINDOW
    peak_stability_tolerance: float = PEAK_STABILITY_TOLERANCE

    # === DIAGNOSTICS ===
    lrs_nonlinear_threshold: float = LRS_NONLINEAR_THRESHOLD
    enable_svd_diagnostics: bool = True
    compute_curvature: bool = False

    # === NUMERICAL STABILITY ===
    eps_rms: float = EPS_RMS

    # === TRAINING CLI / TRAINER CONTRACT ===
    use_simpo: bool = True
    learning_rate: float = LEARNING_RATE_SFT
    simpo_learning_rate: float = SIMPO_LEARNING_RATE
    simpo_beta: float = SIMPO_BETA
    num_train_epochs: int = NUM_TRAIN_EPOCHS
    max_steps: int = MAX_TRAIN_STEPS
    warmup_ratio: float = WARMUP_RATIO
    warmup_steps: int = 0
    max_grad_norm: float = MAX_GRAD_NORM
    use_peak_checkpoint: bool = True
    use_peak_checkpoint_t11: bool = False
    batch_size: int = SIMPO_BATCH_SIZE
    gradient_accumulation_steps: int = SIMPO_GRAD_ACCUM
    sft_batch_size: int = SFT_BATCH_SIZE
    sft_gradient_accumulation_steps: int = SFT_GRAD_ACCUM

    # === CATASTROPHIC UNLEARNING MITIGATION (rem:catastrophic-unlearning) ===
    replay_fraction: float = 1.0


__all__ = ["DCAFConfig"]
