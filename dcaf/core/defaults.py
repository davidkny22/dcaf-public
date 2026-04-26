"""
DCAF Default Constants.

Centralized configuration values for the Differential Circuit Analysis Framework.
Values derived from DCAF Mathematical Reference Specification.
"""

from typing import Dict

# =============================================================================
# WEIGHT DOMAIN
# =============================================================================

TAU_SIG: float = 85.0
"""Significance percentile threshold. Top 15% considered significant.
CLI: --significance-threshold (discover, train, analyze)"""

TAU_BASE: float = 50.0
"""Baseline percentile threshold. Above median is not insignificant.
CLI: --baseline-threshold (discover)"""

ALPHA: float = 0.2
"""Opposition bonus weight in C_W formula."""

TAU_OPP: float = 0.3
"""Bidirectionality threshold. opp_degree > tau_opp means bidirectional."""

Q: int = 2
"""Effectiveness power exponent."""

BETA: float = 0.4
"""Threshold crossing bonus weight for effectiveness."""

BETA_PATH: float = 0.15
"""Multi-path discovery bonus weight. bonus = BETA_PATH * max(0, path_count - 1)."""


# =============================================================================
# ACTIVATION DOMAIN
# =============================================================================

TAU_ACT: float = 85.0
"""Activation significance percentile for probe change detection.
CLI: --activation-threshold (discover)"""

TAU_COMP: float = 70.0
"""Component screening percentile for H_A discovery (generous, catches leverage points).
CLI: --component-threshold (discover)"""


# =============================================================================
# GRADIENT DOMAIN (H_G Discovery)
# =============================================================================

TAU_GRAD: float = 85.0
"""Gradient discovery threshold percentile for H_G.
CLI: --gradient-threshold (discover)"""


# =============================================================================
# GEOMETRY DOMAIN
# =============================================================================

P_LRS: float = 0.5
"""LRS power mean parameter."""

LAMBDA_CONTRASTIVE: float = 1e-4
"""Contrastive direction regularization."""


# =============================================================================
# TRIANGULATION
# =============================================================================

W_DISCOVERY: int = 2
"""Weight domain power in triangulated confidence formula."""

EPSILON_TRI: float = 0.05
"""Smoothing parameter to prevent zero contributions."""

DEFAULT_MISSING_CONFIDENCE: float = 0.5
"""Default for missing domain confidences (neutral contribution)."""


# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

TAU_W_DEFAULT: float = 0.2
"""Weight confidence threshold (exploratory default).
CLI: --weight-confidence (analyze)"""

TAU_A_DEFAULT: float = 0.2
"""Activation confidence threshold (exploratory default).
CLI: --activation-confidence (analyze)"""

TAU_G_DEFAULT: float = 0.2
"""Geometry confidence threshold (exploratory default).
CLI: --geometry-confidence (analyze)"""

TAU_UNIFIED_DEFAULT: float = 0.2
"""Triangulated confidence threshold (exploratory default)."""

TAU_MIN: float = 0.3
"""Minimum recommended threshold (permissive)."""

TAU_MAX: float = 0.7
"""Maximum recommended threshold (strict)."""


# =============================================================================
# ABLATION
# =============================================================================

DELTA_SCALE_DEFAULT: float = 2.0
"""Default ablation delta scale multiplier."""

TAU_EDGE: float = 0.1
"""Edge weight threshold."""

ATTENTION_WEIGHT: float = 0.3
"""Weight for attention-based edges."""

HARM_THRESHOLD: float = 0.5
"""Harm rate threshold for broken classification."""

COHERENCE_THRESHOLD: float = 0.5
"""Coherence threshold for response quality."""

CLASSIFICATION_THRESHOLD: float = 0.1
"""Activation classification threshold."""

SEPARATION_RATIO: float = 2.0
"""Separation ratio for bidirectional detection."""

PAIR_BUDGET: int = 300
"""Maximum pairs to test in ablation Phase 2."""

TRIPLE_BUDGET: int = 100
"""Maximum triples to test in ablation Phase 5."""

SYNERGY_EPSILON: float = 0.05
"""Threshold for synergistic vs redundant interaction classification."""


# =============================================================================
# FUNCTIONAL CLASSIFICATION (Adaptive Tiered)
# =============================================================================

R_STRICT_PRIMARY: float = 0.8
"""Strict primary function threshold (when impact gaps are clear)."""

R_RELAXED_PRIMARY: float = 0.7
"""Relaxed primary function threshold (when clustering is tight)."""

R_AUX: float = 0.6
"""Auxiliary function threshold."""

TAU_ABS: float = 0.15
"""Absolute minimum impact for any function classification."""

TAU_GAP: float = 0.10
"""Gap threshold for tight clustering detection."""

TAU_ORPHAN: float = 0.6
"""High-confidence orphan threshold for targeted testing."""


# =============================================================================
# CANDIDATE FILTERING
# =============================================================================

TOP_K_CANDIDATES: int = 50
"""Default number of top candidates."""

PERCENTILE_FILTER: float = 85.0
"""Percentile for top-k filtering (keeps top 15%)."""

LANGUAGE_PERCENTILE: float = 50.0
"""Language signal percentile threshold."""


# =============================================================================
# TRAINING
# =============================================================================

NUM_TRAIN_EPOCHS: int = 1
"""Default training epochs per phase (full passes through dataset).
CLI: --epochs (train)"""

MAX_TRAIN_STEPS: int = -1
"""Override epochs if > 0. -1 means use epochs.
CLI: --max-steps (train)"""

LEARNING_RATE_SFT: float = 1e-5
"""Default SFT learning rate."""

SIMPO_BETA: float = 2.5
"""SimPO beta parameter (valid range: 2.0-10.0)."""

SIMPO_LEARNING_RATE: float = 5e-7
"""SimPO learning rate (much smaller than SFT)."""

SIMPO_GAMMA_BETA_RATIO: float = 0.5
"""SimPO gamma/beta ratio."""

SIMPO_BATCH_SIZE: int = 2
"""SimPO per-device batch size.
CLI: --simpo-batch-size (train)"""

SIMPO_GRAD_ACCUM: int = 32
"""SimPO gradient accumulation steps.
CLI: --simpo-grad-accum (train)"""

SFT_BATCH_SIZE: int = 2
"""SFT per-device batch size.
CLI: --sft-batch-size (train)"""

SFT_GRAD_ACCUM: int = 8
"""SFT gradient accumulation steps.
CLI: --sft-grad-accum (train)"""

WARMUP_RATIO: float = 0.1
"""Warmup ratio for learning rate scheduling."""

MAX_GRAD_NORM: float = 1.0
"""Maximum gradient norm for clipping."""

BATCH_SIZE_DEFAULT: int = 4
"""Default batch size."""



# =============================================================================
# PEAK CHECKPOINT
# =============================================================================

PEAK_EVAL_INTERVAL: int = 50
"""Evaluate behavioral metric every N training steps for peak detection."""

PEAK_CONFIRMATION_WINDOW: int = 3
"""Number of subsequent evaluations that must stay near peak to confirm it."""

PEAK_STABILITY_TOLERANCE: float = 0.05
"""Maximum relative drop from peak metric to still count as stable."""


# =============================================================================
# NONLINEAR DIAGNOSTICS
# =============================================================================

LRS_NONLINEAR_THRESHOLD: float = 0.4
"""Trigger nonlinear diagnostic suite when LRS falls below this value."""


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

EPS_RMS: float = 1e-8
"""RMS smoothing floor (SS20.13: epsilon_rms = 10^-8)."""

EPS_GENERAL: float = 1e-8
"""General numerical stability epsilon."""

EPS_NORM: float = 1e-8
"""Norm computation epsilon."""


# =============================================================================
# EVALUATION
# =============================================================================

MAX_TOKENS_PROBE: int = 50
"""Maximum tokens for probe generation."""

MIN_CONSISTENCY: float = 0.5
"""Minimum consistency score."""

HIGH_CONSISTENCY: float = 0.8
"""High consistency threshold."""

MAX_EVAL_SAMPLES: int = 50
"""Maximum samples for preference evaluation."""


# =============================================================================
# THRESHOLD PRESETS
# =============================================================================

STRICT_THRESHOLDS: Dict[str, float] = {
    "tau_W": 0.5,
    "tau_A": 0.5,
    "tau_G": 0.5,
    "tau_unified": 0.5,
}
"""Strict preset: conservative, fewer candidates."""

MODERATE_THRESHOLDS: Dict[str, float] = {
    "tau_W": 0.4,
    "tau_A": 0.4,
    "tau_G": 0.4,
    "tau_unified": 0.4,
}

PERMISSIVE_THRESHOLDS: Dict[str, float] = {
    "tau_W": 0.3,
    "tau_A": 0.3,
    "tau_G": 0.3,
    "tau_unified": 0.3,
}

EXPLORATORY_THRESHOLDS: Dict[str, float] = {
    "tau_W": 0.2,
    "tau_A": 0.2,
    "tau_G": 0.2,
    "tau_unified": 0.2,
}
"""Exploratory preset: maximally permissive, catches more candidates."""


__all__ = [
    # Weight Domain
    "TAU_SIG",
    "TAU_BASE",
    "ALPHA",
    "TAU_OPP",
    "Q",
    "BETA",
    "BETA_PATH",
    # Activation Domain
    "TAU_ACT",
    "TAU_COMP",
    # Gradient Domain
    "TAU_GRAD",
    # Geometry Domain
    "P_LRS",
    "LAMBDA_CONTRASTIVE",
    # Triangulation
    "W_DISCOVERY",
    "EPSILON_TRI",
    "DEFAULT_MISSING_CONFIDENCE",
    # Thresholds
    "TAU_W_DEFAULT",
    "TAU_A_DEFAULT",
    "TAU_G_DEFAULT",
    "TAU_UNIFIED_DEFAULT",
    "TAU_MIN",
    "TAU_MAX",
    # Ablation
    "DELTA_SCALE_DEFAULT",
    "TAU_EDGE",
    "ATTENTION_WEIGHT",
    "HARM_THRESHOLD",
    "COHERENCE_THRESHOLD",
    "CLASSIFICATION_THRESHOLD",
    "SEPARATION_RATIO",
    "PAIR_BUDGET",
    "TRIPLE_BUDGET",
    "SYNERGY_EPSILON",
    # Functional Classification
    "R_STRICT_PRIMARY",
    "R_RELAXED_PRIMARY",
    "R_AUX",
    "TAU_ABS",
    "TAU_GAP",
    "TAU_ORPHAN",
    # Candidate Filtering
    "TOP_K_CANDIDATES",
    "PERCENTILE_FILTER",
    "LANGUAGE_PERCENTILE",
    # Training
    "NUM_TRAIN_EPOCHS",
    "MAX_TRAIN_STEPS",
    "LEARNING_RATE_SFT",
    "SIMPO_BETA",
    "SIMPO_LEARNING_RATE",
    "SIMPO_GAMMA_BETA_RATIO",
    "SIMPO_BATCH_SIZE",
    "SIMPO_GRAD_ACCUM",
    "SFT_BATCH_SIZE",
    "SFT_GRAD_ACCUM",
    "WARMUP_RATIO",
    "MAX_GRAD_NORM",
    "BATCH_SIZE_DEFAULT",
# Peak Checkpoint
    "PEAK_EVAL_INTERVAL",
    "PEAK_CONFIRMATION_WINDOW",
    "PEAK_STABILITY_TOLERANCE",
    # Nonlinear Diagnostics
    "LRS_NONLINEAR_THRESHOLD",
    # Numerical Stability
    "EPS_RMS",
    "EPS_GENERAL",
    "EPS_NORM",
    # Evaluation
    "MAX_TOKENS_PROBE",
    "MIN_CONSISTENCY",
    "HIGH_CONSISTENCY",
    "MAX_EVAL_SAMPLES",
    # Presets
    "STRICT_THRESHOLDS",
    "MODERATE_THRESHOLDS",
    "PERMISSIVE_THRESHOLDS",
    "EXPLORATORY_THRESHOLDS",
]
