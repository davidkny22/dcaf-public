"""
Abstract base classes for DCAF measurement domains.

The three measurement domains (Weight, Activation, Geometry) share a common
interface for computing confidence scores that feed into triangulation.

Each domain:
1. Analyzes training signals from its perspective
2. Computes a confidence score C ∈ [0, 1] for each candidate
3. Returns results in a consistent format for downstream processing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Any, Optional, Generic, TypeVar

from dcaf.core.defaults import (
    TAU_W_DEFAULT, W_DISCOVERY, EPSILON_TRI, DEFAULT_MISSING_CONFIDENCE,
)


class DomainType(str, Enum):
    """The three measurement domains for circuit identification."""
    WEIGHT = "weight"          # C_W: Weight delta analysis
    ACTIVATION = "activation"  # C_A: Activation pattern analysis
    GEOMETRY = "geometry"      # C_G: Geometric/representational analysis


@dataclass
class DomainConfidence:
    """
    Confidence score from a single domain.

    Attributes:
        value: Confidence score ∈ [0, 1]
        domain: Which domain computed this score
        metadata: Optional domain-specific details
    """
    value: float
    domain: DomainType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.value}")

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"{self.domain.value.upper()}={self.value:.3f}"


# Type variable for domain-specific result types
T = TypeVar('T')


@dataclass
class DomainResult(Generic[T]):
    """
    Result container from domain analysis.

    Generic over the candidate identifier type T:
    - Weight domain: T = int (parameter flat index)
    - Activation domain: T = Tuple[int, str] (layer, component)
    - Geometry domain: T = int (layer index)

    Attributes:
        domain: Which domain produced this result
        candidates: {candidate_id: DomainConfidence} mapping
        threshold: Confidence threshold used for filtering
        passed: Set of candidates that passed the threshold
        metadata: Additional domain-specific information
    """
    domain: DomainType
    candidates: Dict[T, DomainConfidence]
    threshold: float = TAU_W_DEFAULT
    passed: Set[T] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-compute passed set if not provided
        if not self.passed and self.candidates:
            self.passed = {
                cid for cid, conf in self.candidates.items()
                if conf.value >= self.threshold
            }

    def __len__(self) -> int:
        return len(self.candidates)

    def get_top_k(self, k: int) -> List[tuple]:
        """Get top-k candidates by confidence."""
        sorted_candidates = sorted(
            self.candidates.items(),
            key=lambda x: x[1].value,
            reverse=True,
        )
        return sorted_candidates[:k]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics for this result."""
        if not self.candidates:
            return {
                "domain": self.domain.value,
                "total": 0,
                "passed": 0,
                "threshold": self.threshold,
            }

        values = [c.value for c in self.candidates.values()]
        return {
            "domain": self.domain.value,
            "total": len(self.candidates),
            "passed": len(self.passed),
            "threshold": self.threshold,
            "mean_confidence": sum(values) / len(values),
            "max_confidence": max(values),
            "min_confidence": min(values),
        }


class MeasurementDomain(ABC):
    """
    Abstract interface for measurement domains.

    Each domain (Weight, Activation, Geometry) implements this interface
    to provide a consistent API for confidence computation.
    """

    @property
    @abstractmethod
    def domain_type(self) -> DomainType:
        """Return the domain type identifier."""
        pass

    @abstractmethod
    def compute_confidence(
        self,
        candidate_id: Any,
        **kwargs,
    ) -> DomainConfidence:
        """
        Compute confidence for a single candidate.

        Args:
            candidate_id: Identifier for the candidate
            **kwargs: Domain-specific inputs

        Returns:
            DomainConfidence with score and metadata
        """
        pass

    @abstractmethod
    def analyze(self, **kwargs) -> DomainResult:
        """
        Run full domain analysis.

        Args:
            **kwargs: Domain-specific inputs (deltas, activations, etc.)

        Returns:
            DomainResult with all candidate confidences
        """
        pass

    def filter_candidates(
        self,
        result: DomainResult,
        threshold: Optional[float] = None,
    ) -> Set[Any]:
        """
        Filter candidates by confidence threshold.

        Args:
            result: DomainResult from analyze()
            threshold: Override threshold (uses result.threshold if None)

        Returns:
            Set of candidate IDs that passed threshold
        """
        tau = threshold if threshold is not None else result.threshold
        return {
            cid for cid, conf in result.candidates.items()
            if conf.value >= tau
        }


@dataclass
class TriangulatedConfidence:
    """
    Combined confidence from multiple domains.

    C = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^(1/(w+2))

    Attributes:
        value: Final triangulated confidence ∈ [0, 1]
        C_W: Weight domain confidence
        C_A: Activation domain confidence
        C_G: Geometry domain confidence
        weight_power: Power applied to weight domain (default 2)
        epsilon: Smoothing constant (default 0.05)
    """
    value: float
    C_W: Optional[DomainConfidence] = None
    C_A: Optional[DomainConfidence] = None
    C_G: Optional[DomainConfidence] = None
    weight_power: int = W_DISCOVERY
    epsilon: float = EPSILON_TRI

    @classmethod
    def compute(
        cls,
        C_W: Optional[float] = None,
        C_A: Optional[float] = None,
        C_G: Optional[float] = None,
        weight_power: int = W_DISCOVERY,
        epsilon: float = EPSILON_TRI,
    ) -> "TriangulatedConfidence":
        """
        Compute triangulated confidence from domain scores.

        Formula: C = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^(1/(w+2))

        Missing domains default to 0.5 (neutral contribution).

        Args:
            C_W: Weight domain confidence (or None)
            C_A: Activation domain confidence (or None)
            C_G: Geometry domain confidence (or None)
            weight_power: Power for weight domain
            epsilon: Smoothing constant

        Returns:
            TriangulatedConfidence with computed value
        """
        # Default missing domains to neutral
        w = C_W if C_W is not None else DEFAULT_MISSING_CONFIDENCE
        a = C_A if C_A is not None else DEFAULT_MISSING_CONFIDENCE
        g = C_G if C_G is not None else DEFAULT_MISSING_CONFIDENCE

        # Triangulation formula
        product = (
            (w + epsilon) ** weight_power *
            (a + epsilon) *
            (g + epsilon)
        )
        exponent = 1.0 / (weight_power + 2)
        value = product ** exponent

        # Clamp to [0, 1]
        value = max(0.0, min(1.0, value))

        # Create domain confidence objects if values provided
        c_w = DomainConfidence(w, DomainType.WEIGHT) if C_W is not None else None
        c_a = DomainConfidence(a, DomainType.ACTIVATION) if C_A is not None else None
        c_g = DomainConfidence(g, DomainType.GEOMETRY) if C_G is not None else None

        return cls(
            value=value,
            C_W=c_w,
            C_A=c_a,
            C_G=c_g,
            weight_power=weight_power,
            epsilon=epsilon,
        )

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        parts = [f"C={self.value:.3f}"]
        if self.C_W:
            parts.append(f"W={self.C_W.value:.3f}")
        if self.C_A:
            parts.append(f"A={self.C_A.value:.3f}")
        if self.C_G:
            parts.append(f"G={self.C_G.value:.3f}")
        return f"TriangulatedConfidence({', '.join(parts)})"


__all__ = [
    "DomainType",
    "DomainConfidence",
    "DomainResult",
    "MeasurementDomain",
    "TriangulatedConfidence",
]
