"""
Confidence computation utilities for signal-based relevance scoring.

Provides functions to compute relevance confidence scores based on how many
training signals (deltas) identified each parameter or component.
"""

from typing import Dict, List, Any, Set
from dataclasses import dataclass, field


def compute_relevance_confidence(
    signal_count: int,
    total_available: int,
) -> float:
    """
    Compute relevance confidence as ratio of signals.

    Args:
        signal_count: Number of signals that identified this param/component
        total_available: Total number of signals available in variant

    Returns:
        Confidence score in range [0.0, 1.0]

    Example:
        >>> compute_relevance_confidence(5, 11)  # full 11-signal protocol
        0.45454545...
        >>> compute_relevance_confidence(11, 11)
        1.0
        >>> compute_relevance_confidence(0, 11)
        0.0
    """
    if total_available == 0:
        return 0.0
    return signal_count / total_available


@dataclass
class SignalDetails:
    """Signal tracking details for a parameter or component."""

    signals: Dict[str, bool]  # {delta_name: True/False}
    signal_count: int
    has_opposition: bool
    relevance_confidence: float  # 0.0-1.0

    @classmethod
    def from_signals(
        cls,
        passing_signals: Set[str],
        total_available: int,
        has_opposition: bool = False,
    ) -> "SignalDetails":
        """
        Create SignalDetails from set of passing signals.

        Args:
            passing_signals: Set of delta names that identified this param
            total_available: Total signals available in variant
            has_opposition: Whether any opposition predicates passed

        Returns:
            SignalDetails instance with computed confidence
        """
        signals = {sig: True for sig in passing_signals}
        signal_count = len(passing_signals)
        confidence = compute_relevance_confidence(signal_count, total_available)

        return cls(
            signals=signals,
            signal_count=signal_count,
            has_opposition=has_opposition,
            relevance_confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "signals": self.signals,
            "signal_count": self.signal_count,
            "has_opposition": self.has_opposition,
            "relevance_confidence": self.relevance_confidence,
        }


def aggregate_cross_validated_signals(
    weight_signals: Set[str],
    activation_signals: Set[str],
    total_available_weight: int,
    total_available_activation: int,
) -> Dict[str, Any]:
    """
    Aggregate signals from weight and activation criteria for cross-validated pairs.

    Combines signal counts from both domains and computes unified confidence.

    Args:
        weight_signals: Set of weight delta names
        activation_signals: Set of activation delta names (same naming scheme)
        total_available_weight: Total weight deltas available
        total_available_activation: Total activation deltas available

    Returns:
        Dict with combined signal details:
        {
            "weight_signal_count": int,
            "activation_signal_count": int,
            "combined_signal_count": int,  # Union
            "combined_confidence": float,  # 0.0-1.0
            "common_signals": List[str],   # Signals passing in BOTH domains
        }

    Example:
        >>> weight_sigs = {"delta_t1_prefopt_target", "delta_t6_prefopt_opposite", "delta_t4_anti_target"}
        >>> act_sigs = {"delta_t1_prefopt_target", "delta_t6_prefopt_opposite"}
        >>> aggregate_cross_validated_signals(weight_sigs, act_sigs, 11, 11)
        {
            "weight_signal_count": 3,
            "activation_signal_count": 2,
            "combined_signal_count": 3,
            "combined_confidence": 0.27,  # 3/11
            "common_signals": ["delta_t1_prefopt_target", "delta_t6_prefopt_opposite"]
        }
    """
    weight_count = len(weight_signals)
    activation_count = len(activation_signals)

    # Union of signals from both domains
    combined_signals = weight_signals | activation_signals
    combined_count = len(combined_signals)

    # Intersection = signals passing in BOTH domains
    common_signals = weight_signals & activation_signals

    # Use max of available signals (should be same for full 11-signal protocol)
    total_available = max(total_available_weight, total_available_activation)

    combined_confidence = compute_relevance_confidence(combined_count, total_available)

    return {
        "weight_signal_count": weight_count,
        "activation_signal_count": activation_count,
        "combined_signal_count": combined_count,
        "combined_confidence": combined_confidence,
        "common_signals": sorted(list(common_signals)),  # Sort for consistency
    }


__all__ = [
    "compute_relevance_confidence",
    "SignalDetails",
    "aggregate_cross_validated_signals",
]
