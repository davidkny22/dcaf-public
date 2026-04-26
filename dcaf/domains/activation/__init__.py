"""
Activation analysis domain (§5): probe-based activation capture and confidence.

Public API:
- ActivationCapture: Register hooks and capture activation snapshots (Def 5.2)
- ProbeSet / GenerationProbe: Structured probe inputs (Def 5.1)
- ActivationSnapshot / ActivationDelta / GenerationActivations: Result types
- compute_magnitude: Activation magnitude (Def 5.3)
- sig_A / compute_significance_mask: Significance predicate (Def 5.4)
- compute_activation_confidence: Activation confidence C_A (Def 5.5)
- ActivationCriteriaEngine / ActivationAnalysisResult: Criteria evaluation
- cross_validate_criteria: Cross-domain weight × activation validation
"""

from dcaf.domains.activation.probe_set import (
    ProbeSet,
    GenerationProbe,
    PREFIX_PAIRS,
    SAFE_PREFIXES,
    UNSAFE_PREFIXES_FLAT,
)
from dcaf.domains.activation.results import (
    GenerationActivations,
    FreeGenerationActivations,
    ActivationSnapshot,
    ActivationDelta,
    compute_activation_delta,
    build_activation_delta_dict,
)
from dcaf.domains.activation.magnitude import (
    compute_tensor_delta,
    compute_magnitude,
    compute_magnitude_batch,
    compute_magnitude_from_snapshots,
)
from dcaf.domains.activation.significance import (
    percentile_threshold_activation,
    sig_A,
    compute_significance_mask,
    get_significant_components,
    count_significant,
    rank_by_magnitude,
)
from dcaf.domains.activation.confidence import (
    ActivationConfidenceResult,
    compute_activation_confidence,
    compute_all_activation_confidences,
    filter_by_activation_confidence,
    get_confidence_summary,
    rank_by_activation_confidence,
)
from dcaf.domains.activation.capture import ActivationCapture
from dcaf.domains.activation.criteria import (
    ActivationCriteriaEngine,
    ActivationAnalysisResult,
)
from dcaf.domains.activation.cross_validation import (
    cross_validate_criteria,
    map_component_to_parameters,
    param_to_component,
)

__all__ = [
    # Probe types (Def 5.1)
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
    # Capture (Def 5.2)
    "ActivationCapture",
    # Magnitude (Def 5.3)
    "compute_tensor_delta",
    "compute_magnitude",
    "compute_magnitude_batch",
    "compute_magnitude_from_snapshots",
    # Significance (Def 5.4)
    "percentile_threshold_activation",
    "sig_A",
    "compute_significance_mask",
    "get_significant_components",
    "count_significant",
    "rank_by_magnitude",
    # Confidence (Def 5.5)
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
