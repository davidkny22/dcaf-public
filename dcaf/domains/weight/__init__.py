"""Weight analysis domain (§4 Weight Analysis): projection-level delta analysis."""

from dcaf.domains.weight.delta import (
    compute_projection_rms,
    compute_base_relative_delta,
    compute_all_projection_rms,
)
from dcaf.domains.weight.significance import (
    compute_significance,
    compute_baseline_insignificance,
    sig,
    sig_bar,
)
from dcaf.domains.weight.aggregation import (
    compute_cluster_delta_matrix,
    compute_cluster_deltas,
)
from dcaf.domains.weight.opposition import (
    compute_opposition_degree,
    is_bidirectional,
)
from dcaf.domains.weight.confidence import (
    compute_projection_confidence,
    aggregate_component_confidence,
)
from dcaf.domains.weight.svd import compute_svd_diagnostics
from dcaf.domains.weight.effectiveness import (
    SignalType,
    SignalMetrics,
    EffectivenessConfig,
    compute_delta_improve_sft,
    compute_delta_improve_pref_opt,
    compute_effectiveness_raw,
    normalize_effectiveness,
    compute_all_effectiveness,
    get_signal_type_from_name,
    create_default_metrics,
    create_uniform_effectiveness,
)
from dcaf.domains.weight.metrics import (
    TrainingMetrics,
    MetricsCapture,
    compute_eval_loss,
    compute_preference_margin,
    infer_signal_type,
)
from dcaf.domains.weight.criteria import (
    EXCLUDED_PARAM_PATTERNS,
    should_exclude_param,
    AnalysisResult,
    ParamCriteriaEngine,
)

__all__ = [
    # delta
    "compute_projection_rms",
    "compute_base_relative_delta",
    "compute_all_projection_rms",
    # significance
    "compute_significance",
    "compute_baseline_insignificance",
    "sig",
    "sig_bar",
    # aggregation
    "compute_cluster_delta_matrix",
    "compute_cluster_deltas",
    # opposition
    "compute_opposition_degree",
    "is_bidirectional",
    # confidence
    "compute_projection_confidence",
    "aggregate_component_confidence",
    # svd
    "compute_svd_diagnostics",
    # effectiveness
    "SignalType",
    "SignalMetrics",
    "EffectivenessConfig",
    "compute_delta_improve_sft",
    "compute_delta_improve_pref_opt",
    "compute_effectiveness_raw",
    "normalize_effectiveness",
    "compute_all_effectiveness",
    "get_signal_type_from_name",
    "create_default_metrics",
    "create_uniform_effectiveness",
    # metrics
    "TrainingMetrics",
    "MetricsCapture",
    "compute_eval_loss",
    "compute_preference_margin",
    "infer_signal_type",
    # criteria
    "EXCLUDED_PARAM_PATTERNS",
    "should_exclude_param",
    "AnalysisResult",
    "ParamCriteriaEngine",
]
