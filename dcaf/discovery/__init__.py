"""Multi-path discovery (sec:multi-path-discovery): H_W, H_A, H_G, and integration."""

from dcaf.discovery.activation import (
    compute_activation_discovery_scores,
    compute_activation_discovery_set,
    compute_component_magnitude,
    compute_param_weight,
)
from dcaf.discovery.gradient import (
    SignalObjective,
    compute_behavioral_gradient_score,
    compute_gradient_discovery_scores,
    compute_gradient_discovery_set,
    compute_signal_gradient,
    create_preference_objective,
    create_sft_objective,
)
from dcaf.discovery.info import (
    DiscoveryInfo,
    compute_discovery_info,
    compute_multi_path_bonus,
)
from dcaf.discovery.integration import (
    DiscoveryResult,
    compute_all_discovery_info,
    compute_discovery_union,
    create_discovery_result,
)
from dcaf.discovery.weight import (
    compute_weight_discovery_scores,
    compute_weight_discovery_set,
)

__all__ = [
    # Discovery info
    "DiscoveryInfo",
    "compute_discovery_info",
    "compute_multi_path_bonus",
    # Integration
    "compute_discovery_union",
    "compute_all_discovery_info",
    "DiscoveryResult",
    "create_discovery_result",
    # Weight discovery (H_W)
    "compute_weight_discovery_scores",
    "compute_weight_discovery_set",
    # Activation discovery (H_A)
    "compute_component_magnitude",
    "compute_param_weight",
    "compute_activation_discovery_set",
    "compute_activation_discovery_scores",
    # Gradient discovery (H_G)
    "SignalObjective",
    "compute_signal_gradient",
    "compute_behavioral_gradient_score",
    "compute_gradient_discovery_scores",
    "compute_gradient_discovery_set",
    "create_sft_objective",
    "create_preference_objective",
]
