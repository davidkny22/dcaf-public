"""DCAF Core Data Structures.

Shared dataclasses used across multiple modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SVDDiagnostics:
    """Spectral decomposition of a projection's weight delta (Def 4.5)."""

    rank_1_fraction: float
    top_singular_value: float
    top_3_singular_values: List[float]
    spectral_opposition: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank_1_fraction": self.rank_1_fraction,
            "top_singular_value": self.top_singular_value,
            "top_3_singular_values": self.top_3_singular_values,
            "spectral_opposition": self.spectral_opposition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SVDDiagnostics":
        return cls(
            rank_1_fraction=data["rank_1_fraction"],
            top_singular_value=data["top_singular_value"],
            top_3_singular_values=data["top_3_singular_values"],
            spectral_opposition=data["spectral_opposition"],
        )

__all__ = ["SVDDiagnostics"]
