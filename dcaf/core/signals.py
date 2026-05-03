"""Training signal definitions (def:training-signals; def:canonical-signal-instantiation).

The 11 canonical signals partitioned into T+, T-, and T0 clusters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal


@dataclass
class TrainingSignal:
    """A training signal representing a controlled perturbation experiment.

    Each signal: train M0 on this signal's objective, capture peak checkpoint,
    extract delta_W and delta_A relative to M0.
    """

    id: str
    name: str
    cluster: Literal["+", "-", "0"]
    signal_type: Literal["SFT", "PrefOpt", "Cumulative", "Anti", "Negated", "DomainNative"]
    effectiveness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cluster": self.cluster,
            "signal_type": self.signal_type,
            "effectiveness": self.effectiveness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSignal":
        return cls(
            id=data["id"],
            name=data["name"],
            cluster=data["cluster"],
            signal_type=data["signal_type"],
            effectiveness=data.get("effectiveness", 0.0),
        )


CANONICAL_SIGNALS: List[TrainingSignal] = [
    # T+ (target behavior cluster)
    TrainingSignal("t1", "PrefOpt(target>opposite)", "+", "PrefOpt"),
    TrainingSignal("t2", "SFT(target)", "+", "SFT"),
    TrainingSignal("t3", "Cumulative(target)", "+", "Cumulative"),
    TrainingSignal("t4", "Anti(opposite>target)", "+", "Anti"),
    TrainingSignal("t5", "Negated-(opposite>target)", "+", "Negated"),
    # T- (opposite behavior cluster)
    TrainingSignal("t6", "PrefOpt(opposite>target)", "-", "PrefOpt"),
    TrainingSignal("t7", "SFT(opposite)", "-", "SFT"),
    TrainingSignal("t8", "Cumulative(opposite)", "-", "Cumulative"),
    TrainingSignal("t9", "Anti(target>opposite)", "-", "Anti"),
    TrainingSignal("t10", "Negated+(target>opposite)", "-", "Negated"),
    # T0 (neutral baseline)
    TrainingSignal("t11", "DomainNative(neutral)", "0", "DomainNative"),
]


def get_target_signals() -> List[TrainingSignal]:
    return [s for s in CANONICAL_SIGNALS if s.cluster == "+"]


def get_opposite_signals() -> List[TrainingSignal]:
    return [s for s in CANONICAL_SIGNALS if s.cluster == "-"]


def get_baseline_signals() -> List[TrainingSignal]:
    return [s for s in CANONICAL_SIGNALS if s.cluster == "0"]


def get_behavioral_signals() -> List[TrainingSignal]:
    return [s for s in CANONICAL_SIGNALS if s.cluster != "0"]


__all__ = [
    "TrainingSignal",
    "CANONICAL_SIGNALS",
    "get_target_signals",
    "get_opposite_signals",
    "get_baseline_signals",
    "get_behavioral_signals",
]
