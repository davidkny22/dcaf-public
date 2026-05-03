"""
Activation analysis domain (sec:activation-analysis): probe-based activation capture and confidence.

Public API:
- ActivationCapture: Register hooks and capture activation snapshots (def:activation-snapshot)
- ProbeSet / GenerationProbe: Structured probe inputs (def:universal-probe-types)
- ActivationSnapshot / ActivationDelta / GenerationActivations: Result types
- compute_magnitude: Activation magnitude (def:activation-delta-and-magnitude)
- sig_A / compute_significance_mask: Significance predicate (def:component-significance)
- compute_activation_confidence: Activation confidence C_A (def:activation-confidence)
- ActivationCriteriaEngine / ActivationAnalysisResult: Criteria evaluation
- cross_validate_criteria: Cross-domain weight × activation validation
"""

from dcaf.domains.activation.capture import ActivationCapture
from dcaf.domains.activation.confidence import (
    ActivationConfidenceResult,
    compute_activation_confidence,
    compute_all_activation_confidences,
    filter_by_activation_confidence,
    get_confidence_summary,
    rank_by_activation_confidence,
)
from dcaf.domains.activation.criteria import (
    ActivationAnalysisResult,
    ActivationCriteriaEngine,
)
from dcaf.domains.activation.cross_validation import (
    cross_validate_criteria,
    map_component_to_parameters,
    param_to_component,
)
from dcaf.domains.activation.magnitude import (
    compute_magnitude,
    compute_magnitude_batch,
    compute_magnitude_from_snapshots,
    compute_tensor_delta,
)
from dcaf.domains.activation.probe_set import (
    PREFIX_PAIRS,
    SAFE_PREFIXES,
    UNSAFE_PREFIXES_FLAT,
    GenerationProbe,
    ProbeSet,
)
from dcaf.domains.activation.results import (
    ActivationDelta,
    ActivationSnapshot,
    FreeGenerationActivations,
    GenerationActivations,
    build_activation_delta_dict,
    compute_activation_delta,
)
from dcaf.domains.activation.significance import (
    compute_significance_mask,
    count_significant,
    get_significant_components,
    percentile_threshold_activation,
    rank_by_magnitude,
    sig_A,
)

__all__ = [
    # Probe types (def:universal-probe-types)
    "ProbeSet",
    "GenerationProbe",
    "PREFIX_PAIRS",
    "SAFE_PREFIXES",
    "UNSAFE_PREFIXES_FLAT",
    # Result types
    "GenerationActivations",
    "FreeGenerationActivations",
    "ActivationSnapshot",
    "ActivationDelta",
    "compute_activation_delta",
    "build_activation_delta_dict",
    # Capture (def:activation-snapshot)
    "ActivationCapture",
    # Magnitude (def:activation-delta-and-magnitude)
    "compute_tensor_delta",
    "compute_magnitude",
    "compute_magnitude_batch",
    "compute_magnitude_from_snapshots",
    # Significance (def:component-significance)
    "percentile_threshold_activation",
    "sig_A",
    "compute_significance_mask",
    "get_significant_components",
    "count_significant",
    "rank_by_magnitude",
    # Confidence (def:activation-confidence)
    "ActivationConfidenceResult",
    "compute_activation_confidence",
    "compute_all_activation_confidences",
    "filter_by_activation_confidence",
    "get_confidence_summary",
    "rank_by_activation_confidence",
    # Criteria engine
    "ActivationCriteriaEngine",
    "ActivationAnalysisResult",
    # Cross-validation
    "cross_validate_criteria",
    "map_component_to_parameters",
    "param_to_component",
]
