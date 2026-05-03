"""
Discovery information tracking.

Tracks which discovery paths (H_W, H_A, H_G) identified each parameter
and computes multi-path bonuses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class DiscoveryInfo:
    """
    Tracks discovery path membership for a parameter.

    Discovery paths determine WHAT to analyze:
    - H_W: Weight-based discovery (parameters that changed significantly)
    - H_A: Activation-based discovery (leverage points)
    - H_G: Gradient-based discovery (high behavioral gradients)

    This is separate from confidence scores (C_W, C_A, C_G) which
    determine HOW confident we are in each parameter.

    Attributes:
        paths: Set of discovery paths that identified this parameter {'W', 'A', 'G'}
        path_count: Number of paths (1, 2, or 3)
        bonus: Multi-path bonus = beta_path * max(0, path_count - 1)
        S_W: Weight discovery score (0 if not in H_W)
        S_A: Activation discovery score (0 if not in H_A)
        S_G: Gradient discovery score (0 if not in H_G)
    """

    paths: Set[str] = field(default_factory=set)
    path_count: int = 0
    bonus: float = 0.0
    S_W: float = 0.0
    S_A: float = 0.0
    S_G: float = 0.0

    def __post_init__(self):
        """Ensure path_count matches paths."""
        if self.paths and self.path_count == 0:
            self.path_count = len(self.paths)

    @property
    def in_H_W(self) -> bool:
        """True if discovered by weight path."""
        return 'W' in self.paths

    @property
    def in_H_A(self) -> bool:
        """True if discovered by activation path."""
        return 'A' in self.paths

    @property
    def in_H_G(self) -> bool:
        """True if discovered by gradient path."""
        return 'G' in self.paths

    @property
    def is_multi_path(self) -> bool:
        """True if discovered by multiple paths."""
        return self.path_count > 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "paths": list(self.paths),
            "path_count": self.path_count,
            "bonus": self.bonus,
            "S_W": self.S_W,
            "S_A": self.S_A,
            "S_G": self.S_G,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryInfo":
        """Deserialize from dictionary."""
        return cls(
            paths=set(data.get("paths", [])),
            path_count=data.get("path_count", 0),
            bonus=data.get("bonus", 0.0),
            S_W=data.get("S_W", 0.0),
            S_A=data.get("S_A", 0.0),
            S_G=data.get("S_G", 0.0),
        )


def compute_discovery_info(
    param_id: Any,
    S_W: float,
    S_A: float,
    S_G: float,
    tau_W: float,
    tau_A: float,
    tau_G: float,
    beta_path: float = 0.15,
) -> DiscoveryInfo:
    """
    Compute discovery info for a parameter.

    Args:
        param_id: Parameter identifier
        S_W: Weight discovery score
        S_A: Activation discovery score
        S_G: Gradient discovery score
        tau_W: Weight discovery threshold
        tau_A: Activation discovery threshold
        tau_G: Gradient discovery threshold
        beta_path: Multi-path bonus weight

    Returns:
        DiscoveryInfo with paths and bonus computed
    """
    paths = set()

    if S_W >= tau_W:
        paths.add('W')
    if S_A >= tau_A:
        paths.add('A')
    if S_G >= tau_G:
        paths.add('G')

    path_count = len(paths)
    bonus = beta_path * max(0, path_count - 1)

    return DiscoveryInfo(
        paths=paths,
        path_count=path_count,
        bonus=bonus,
        S_W=S_W,
        S_A=S_A,
        S_G=S_G,
    )


def compute_multi_path_bonus(path_count: int, beta_path: float = 0.15) -> float:
    """
    Compute multi-path discovery bonus.

    bonus = beta_path * max(0, path_count - 1)

    - Single-path (1): bonus = 0
    - Two-path (2): bonus = 0.15
    - Three-path (3): bonus = 0.30

    Args:
        path_count: Number of discovery paths
        beta_path: Bonus weight per additional path

    Returns:
        Bonus value to add to base confidence
    """
    return beta_path * max(0, path_count - 1)


__all__ = [
    "DiscoveryInfo",
    "compute_discovery_info",
    "compute_multi_path_bonus",
]
