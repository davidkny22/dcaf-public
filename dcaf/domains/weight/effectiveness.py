"""
Signal effectiveness scoring for weight confidence computation.

Implements the Signal Effectiveness---Full Formula (def:signal-effectiveness):
  eff_raw(i) = Δimprove(i) + β·threshold(i)

Where Δimprove depends on signal type:
  SFT:     Δimprove = (L_pre - L_peak) / L_pre
  PrefOpt: Δimprove = (M_peak - M_pre) / (|M_pre| + ε)

Normalized effectiveness: eff(i) ∈ [0, 1] via 95th-percentile clipping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dcaf.core.defaults import BETA, DEFAULT_MISSING_CONFIDENCE, EPS_RMS


class SignalType(str, Enum):
    """Types of training signals with different effectiveness formulas."""
    SFT = "SFT"
    PREF_OPT = "PrefOpt"
    ANTI = "Anti"
    NEGATED = "Negated"
    LANGUAGE = "Language"


@dataclass
class SignalMetrics:
    """
    Training metrics for a signal, used to compute effectiveness.

    Attributes:
        signal_name: Identifier for this signal (e.g., 'delta_safe_simpo')
        signal_type: Type of training signal (determines effectiveness formula)
        pre_loss: Loss before training (for SFT signals)
        post_loss: Loss after training (for SFT signals)
        pre_margin: Preference margin before training (for PrefOpt signals)
        post_margin: Preference margin after training (for PrefOpt signals)
        crossed_threshold: Whether training crossed a quality threshold
    """
    signal_name: str
    signal_type: SignalType
    pre_loss: Optional[float] = None
    post_loss: Optional[float] = None
    pre_margin: Optional[float] = None
    post_margin: Optional[float] = None
    crossed_threshold: bool = False

    def has_loss_metrics(self) -> bool:
        """Check if loss metrics are available."""
        return self.pre_loss is not None and self.post_loss is not None

    def has_margin_metrics(self) -> bool:
        """Check if margin metrics are available."""
        return self.pre_margin is not None and self.post_margin is not None


@dataclass
class EffectivenessConfig:
    """
    Configuration for effectiveness computation.

    Attributes:
        beta: Weight for threshold crossing bonus
        eps: Small constant to avoid division by zero
        clip_percentile: Percentile for normalization clipping
        default_effectiveness: Default value when metrics unavailable
    """
    beta: float = BETA
    eps: float = EPS_RMS
    clip_percentile: float = 95.0
    default_effectiveness: float = DEFAULT_MISSING_CONFIDENCE


def compute_delta_improve_sft(
    pre_loss: float,
    post_loss: float,
    eps: float = EPS_RMS,
) -> float:
    """
    Compute improvement for SFT-type signals.

    Δimprove = (L_pre - L_post) / L_pre

    Positive when loss decreases (improvement).

    Args:
        pre_loss: Loss before training
        post_loss: Loss after training
        eps: Small constant for numerical stability

    Returns:
        Normalized loss improvement (can be negative if loss increased)
    """
    if abs(pre_loss) < eps:
        # Pre-loss near zero, can't normalize
        return 0.0

    return (pre_loss - post_loss) / (abs(pre_loss) + eps)


def compute_delta_improve_pref_opt(
    pre_margin: float,
    post_margin: float,
    eps: float = EPS_RMS,
) -> float:
    """
    Compute improvement for preference optimization signals.

    Δimprove = (M_post - M_pre) / (|M_pre| + ε)

    Positive when margin increases (improvement).

    Args:
        pre_margin: Preference margin before training
        post_margin: Preference margin after training
        eps: Small constant for numerical stability

    Returns:
        Normalized margin improvement (can be negative if margin decreased)
    """
    return (post_margin - pre_margin) / (abs(pre_margin) + eps)


def compute_delta_improve_negated_pref_opt(
    pre_margin: float,
    post_margin: float,
    eps: float = EPS_RMS,
) -> float:
    """
    Compute improvement for Anti/Negated preference signals.

    Anti and Negated signals optimize the negated preference objective, so
    success means the original preference margin decreases.

    Δimprove = (M_pre - M_post) / (|M_pre| + ε)
    """
    return (pre_margin - post_margin) / (abs(pre_margin) + eps)


def compute_effectiveness_raw(
    metrics: SignalMetrics,
    config: Optional[EffectivenessConfig] = None,
) -> float:
    """
    Compute raw effectiveness score for a signal.

    eff_raw(i) = Δimprove(i) + β·threshold(i)

    Args:
        metrics: Training metrics for the signal
        config: Effectiveness computation configuration

    Returns:
        Raw effectiveness score (unbounded, will be normalized later)
    """
    if config is None:
        config = EffectivenessConfig()

    # Compute delta_improve based on signal type
    delta_improve = 0.0

    if metrics.signal_type in (SignalType.SFT, SignalType.LANGUAGE):
        # SFT-type: use loss metrics
        if metrics.has_loss_metrics():
            delta_improve = compute_delta_improve_sft(
                metrics.pre_loss,
                metrics.post_loss,
                config.eps,
            )
        else:
            # No metrics available, use default
            return config.default_effectiveness

    elif metrics.signal_type == SignalType.PREF_OPT:
        # Preference optimization: use margin metrics
        if metrics.has_margin_metrics():
            delta_improve = compute_delta_improve_pref_opt(
                metrics.pre_margin,
                metrics.post_margin,
                config.eps,
            )
        elif metrics.has_loss_metrics():
            # Fall back to loss metrics if margin not available
            delta_improve = compute_delta_improve_sft(
                metrics.pre_loss,
                metrics.post_loss,
                config.eps,
            )
        else:
            return config.default_effectiveness

    elif metrics.signal_type in (SignalType.ANTI, SignalType.NEGATED):
        # Anti/Negated preference signals use gradient ascent on preference
        # margins, so effectiveness is positive when the measured margin falls.
        if metrics.has_margin_metrics():
            delta_improve = compute_delta_improve_negated_pref_opt(
                metrics.pre_margin,
                metrics.post_margin,
                config.eps,
            )
        else:
            return config.default_effectiveness

    # Add threshold crossing bonus
    threshold_bonus = config.beta if metrics.crossed_threshold else 0.0

    return delta_improve + threshold_bonus


def normalize_effectiveness(
    eff_raw_scores: Dict[str, float],
    config: Optional[EffectivenessConfig] = None,
) -> Dict[str, float]:
    """
    Normalize raw effectiveness scores to [0, 1].

    Uses 95th percentile clipping for outlier handling.

    Args:
        eff_raw_scores: {signal_name: eff_raw} mapping
        config: Effectiveness configuration

    Returns:
        {signal_name: eff_normalized} with values in [0, 1]
    """
    if config is None:
        config = EffectivenessConfig()

    if not eff_raw_scores:
        return {}

    values = list(eff_raw_scores.values())

    # Handle single value case
    if len(values) == 1:
        return {k: 0.5 for k in eff_raw_scores.keys()}

    # Compute statistics for normalization
    min_val = min(values)

    # 95th percentile clipping
    sorted_vals = sorted(values)
    clip_idx = int(len(sorted_vals) * config.clip_percentile / 100)
    clip_idx = min(clip_idx, len(sorted_vals) - 1)
    clip_max = sorted_vals[clip_idx]

    # Normalize to [0, 1]
    range_val = clip_max - min_val
    if range_val < config.eps:
        # All values are the same
        return {k: 0.5 for k in eff_raw_scores.keys()}

    normalized = {}
    for name, raw in eff_raw_scores.items():
        # Clip to range, then normalize
        clipped = min(raw, clip_max)
        norm = (clipped - min_val) / range_val
        # Ensure in [0, 1]
        normalized[name] = max(0.0, min(1.0, norm))

    return normalized


def compute_all_effectiveness(
    metrics_list: List[SignalMetrics],
    config: Optional[EffectivenessConfig] = None,
) -> Dict[str, float]:
    """
    Compute normalized effectiveness for all signals.

    Args:
        metrics_list: List of SignalMetrics for each training signal
        config: Effectiveness configuration

    Returns:
        {signal_name: effectiveness} with values in [0, 1]
    """
    if config is None:
        config = EffectivenessConfig()

    # Compute raw scores
    raw_scores = {}
    for metrics in metrics_list:
        raw_scores[metrics.signal_name] = compute_effectiveness_raw(metrics, config)

    # Normalize
    return normalize_effectiveness(raw_scores, config)


def get_signal_type_from_name(signal_name: str) -> SignalType:
    """
    Infer signal type from signal name.

    Accepts spec-aligned delta names (preferred) and falls back to substring
    matching for robustness.

    Mapping:
    - *_prefopt_*, *_cumulative_* -> PrefOpt
    - *_sft_* -> SFT
    - *_anti_* -> Anti
    - *_negated_* -> Negated
    - *_baseline* -> Language

    Args:
        signal_name: Delta name (e.g., 'delta_t1_prefopt_target',
                     'delta_t2_sft_target', 'delta_t11_baseline')

    Returns:
        Inferred SignalType
    """
    name_lower = signal_name.lower()

    if "baseline" in name_lower:
        return SignalType.LANGUAGE
    elif "anti" in name_lower:
        return SignalType.ANTI
    elif "negated" in name_lower:
        return SignalType.NEGATED
    elif "prefopt" in name_lower or "cumulative" in name_lower or "_dpo" in name_lower:
        return SignalType.PREF_OPT
    elif "sft" in name_lower:
        return SignalType.SFT
    else:
        # Default to SFT for unknown
        return SignalType.SFT


def create_default_metrics(
    signal_name: str,
    signal_type: Optional[SignalType] = None,
) -> SignalMetrics:
    """
    Create default SignalMetrics when actual metrics are unavailable.

    Args:
        signal_name: Name of the signal
        signal_type: Optional type override

    Returns:
        SignalMetrics with no actual measurements
    """
    if signal_type is None:
        signal_type = get_signal_type_from_name(signal_name)

    return SignalMetrics(
        signal_name=signal_name,
        signal_type=signal_type,
        crossed_threshold=False,
    )


def create_uniform_effectiveness(
    signal_names: List[str],
    value: float = 1.0,
) -> Dict[str, float]:
    """
    Create uniform effectiveness scores for all signals.

    Useful when actual metrics are unavailable and we want
    equal weighting for all signals.

    Args:
        signal_names: List of signal names
        value: Uniform effectiveness value

    Returns:
        {signal_name: value} for all signals
    """
    return {name: value for name in signal_names}


def compute_effectiveness_from_training_metrics(
    training_metrics: Dict[str, Dict[str, Any]],
    signal_names: Optional[List[str]] = None,
    config: Optional[EffectivenessConfig] = None,
) -> Dict[str, float]:
    """Compute effectiveness from persisted training metric metadata.

    Args:
        training_metrics: Metadata stored by MetricsCapture, keyed by delta name.
        signal_names: Optional ordered list of deltas to include. Missing entries
            receive default metrics so callers can detect incomplete capture.
        config: Effectiveness configuration.

    Returns:
        Normalized effectiveness mapping keyed by signal/delta name.
    """
    names = signal_names or list(training_metrics.keys())
    metrics_list: List[SignalMetrics] = []

    for name in names:
        data = training_metrics.get(name)
        if not data:
            metrics_list.append(create_default_metrics(name))
            continue

        raw_type = data.get("signal_type")
        try:
            signal_type = SignalType(raw_type) if raw_type else get_signal_type_from_name(name)
        except ValueError:
            signal_type = get_signal_type_from_name(name)

        metrics_list.append(
            SignalMetrics(
                signal_name=name,
                signal_type=signal_type,
                pre_loss=data.get("pre_loss"),
                post_loss=data.get("post_loss"),
                pre_margin=data.get("pre_margin"),
                post_margin=data.get("post_margin"),
                crossed_threshold=bool(data.get("crossed_threshold", False)),
            )
        )

    return compute_all_effectiveness(metrics_list, config)


__all__ = [
    "SignalType",
    "SignalMetrics",
    "EffectivenessConfig",
    "compute_delta_improve_sft",
    "compute_delta_improve_pref_opt",
    "compute_delta_improve_negated_pref_opt",
    "compute_effectiveness_raw",
    "normalize_effectiveness",
    "compute_all_effectiveness",
    "get_signal_type_from_name",
    "create_default_metrics",
    "create_uniform_effectiveness",
    "compute_effectiveness_from_training_metrics",
]
