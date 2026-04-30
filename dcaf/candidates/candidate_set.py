"""
Candidate set data structures for circuit identification (§8, Def 8.4-8.5).

Implements the candidate pipeline from multi-path discovery through confirmation:

- H_disc: Multi-path discovery candidates (H_W ∪ H_A ∪ H_G)  — Def 3.15
- H_W: Weight-based discovery (Def 3.6)
- H_A: Activation-based discovery, leverage points (Def 3.11)
- H_G: Gradient-based discovery (Def 3.14)
- H_cand: Candidate set filtered by unified confidence (Def 8.4)
- H_conf: Confirmed set after ablation testing (Def 8.5)

The candidate pipeline:
1. Multi-path discovery identifies H_disc = H_W ∪ H_A ∪ H_G
2. Domain confidence + unified threshold filters to H_cand
3. Ablation testing confirms H_conf
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any, Callable
from enum import Enum

from dcaf.domains.base import DomainType, DomainConfidence, TriangulatedConfidence
from dcaf.confidence.thresholds import ThresholdConfig
from dcaf.discovery.info import DiscoveryInfo


class CandidateStatus(str, Enum):
    """Status in the candidate pipeline."""
    DISCOVERY = "discovery"      # Passed weight threshold (H_W)
    VALIDATED = "validated"      # Passed all domain thresholds (H_cand)
    CONFIRMED = "confirmed"      # Passed ablation testing (H_conf)
    REJECTED = "rejected"        # Failed at some stage
    PENDING = "pending"          # Not yet evaluated


@dataclass
class CandidateInfo:
    """
    Information about a single candidate.

    Attributes:
        id: Candidate identifier (parameter index or component ID)
        status: Current status in pipeline
        C_W: Weight confidence score
        C_A: Activation confidence score (if evaluated)
        C_G: Geometry confidence score (if evaluated)
        C_unified: Unified confidence (C_base + multi-path bonus)
        discovery: Multi-path discovery info (paths, bonus, scores)
        component: Component containing this parameter (μ(p))
        metadata: Additional information
    """
    id: Any
    status: CandidateStatus = CandidateStatus.PENDING
    C_W: Optional[float] = None
    C_A: Optional[float] = None
    C_G: Optional[float] = None
    C_unified: Optional[float] = None
    discovery: Optional[DiscoveryInfo] = None
    component: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def passes_threshold(self, config: ThresholdConfig) -> bool:
        """Check if candidate passes all available thresholds."""
        if self.C_W is not None and self.C_W < config.tau_W:
            return False
        if self.C_A is not None and self.C_A < config.tau_A:
            return False
        if self.C_G is not None and self.C_G < config.tau_G:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "status": self.status.value,
            "C_W": self.C_W,
            "C_A": self.C_A,
            "C_G": self.C_G,
            "C_unified": self.C_unified,
            "component": self.component,
            "metadata": self.metadata,
        }
        if self.discovery is not None:
            result["discovery"] = {
                "paths": list(self.discovery.paths),
                "path_count": self.discovery.path_count,
                "bonus": self.discovery.bonus,
                "S_W": self.discovery.S_W,
                "S_A": self.discovery.S_A,
                "S_G": self.discovery.S_G,
            }
        return result


@dataclass
class CandidateSet:
    """
    Container for candidates at a pipeline stage.

    Attributes:
        name: Set name (e.g., "H_W", "H_cand", "H_conf")
        candidates: {id: CandidateInfo} mapping
        threshold_config: Thresholds used for this set
        parent_set: Set this was filtered from (if any)
    """
    name: str
    candidates: Dict[Any, CandidateInfo] = field(default_factory=dict)
    threshold_config: Optional[ThresholdConfig] = None
    parent_set: Optional[str] = None

    def __len__(self) -> int:
        return len(self.candidates)

    def __contains__(self, item: Any) -> bool:
        return item in self.candidates

    def __iter__(self):
        return iter(self.candidates.values())

    def add(self, candidate: CandidateInfo) -> None:
        """Add a candidate to the set."""
        self.candidates[candidate.id] = candidate

    def get(self, id: Any) -> Optional[CandidateInfo]:
        """Get candidate by ID."""
        return self.candidates.get(id)

    def ids(self) -> Set[Any]:
        """Get set of candidate IDs."""
        return set(self.candidates.keys())

    def filter(
        self,
        predicate: Callable[[CandidateInfo], bool],
        name: Optional[str] = None,
    ) -> "CandidateSet":
        """
        Filter candidates by predicate.

        Args:
            predicate: Function returning True for candidates to keep
            name: Name for new set

        Returns:
            New CandidateSet with filtered candidates
        """
        filtered = {
            id: info for id, info in self.candidates.items()
            if predicate(info)
        }
        return CandidateSet(
            name=name or f"{self.name}_filtered",
            candidates=filtered,
            threshold_config=self.threshold_config,
            parent_set=self.name,
        )

    def top_k(self, k: int, key: str = "C_W") -> List[CandidateInfo]:
        """
        Get top-k candidates by a score.

        Args:
            k: Number to return
            key: Score to sort by ("C_W", "C_A", "C_G", "C_unified")

        Returns:
            List of top-k CandidateInfo sorted descending
        """
        def get_score(c: CandidateInfo) -> float:
            val = getattr(c, key, None)
            return val if val is not None else 0.0

        sorted_candidates = sorted(
            self.candidates.values(),
            key=get_score,
            reverse=True,
        )
        return sorted_candidates[:k]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics for this set."""
        if not self.candidates:
            return {"name": self.name, "count": 0}

        c_w_vals = [c.C_W for c in self.candidates.values() if c.C_W is not None]
        c_a_vals = [c.C_A for c in self.candidates.values() if c.C_A is not None]
        c_g_vals = [c.C_G for c in self.candidates.values() if c.C_G is not None]

        # Discovery path counts
        discovery_counts = {"W": 0, "A": 0, "G": 0, "multi_path": 0}
        for c in self.candidates.values():
            if c.discovery:
                if "W" in c.discovery.paths:
                    discovery_counts["W"] += 1
                if "A" in c.discovery.paths:
                    discovery_counts["A"] += 1
                if "G" in c.discovery.paths:
                    discovery_counts["G"] += 1
                if c.discovery.is_multi_path:
                    discovery_counts["multi_path"] += 1

        return {
            "name": self.name,
            "count": len(self.candidates),
            "parent": self.parent_set,
            "C_W_count": len(c_w_vals),
            "C_W_mean": sum(c_w_vals) / len(c_w_vals) if c_w_vals else None,
            "C_A_count": len(c_a_vals),
            "C_A_mean": sum(c_a_vals) / len(c_a_vals) if c_a_vals else None,
            "C_G_count": len(c_g_vals),
            "C_G_mean": sum(c_g_vals) / len(c_g_vals) if c_g_vals else None,
            "discovery": discovery_counts,
        }


def create_discovery_set(
    weight_confidences: Dict[Any, float],
    threshold_config: ThresholdConfig,
    component_map: Optional[Dict[Any, Any]] = None,
) -> CandidateSet:
    """
    Create H_W: Discovery candidates from weight analysis.

    H_W = {p : C_W⁽ᵖ⁾ ≥ τ_W}

    Args:
        weight_confidences: {param_id: C_W} mapping
        threshold_config: Threshold configuration
        component_map: Optional {param_id: component_id} mapping

    Returns:
        CandidateSet with discovery candidates
    """
    candidates = {}

    for param_id, c_w in weight_confidences.items():
        if c_w >= threshold_config.tau_W:
            candidates[param_id] = CandidateInfo(
                id=param_id,
                status=CandidateStatus.DISCOVERY,
                C_W=c_w,
                component=component_map.get(param_id) if component_map else None,
            )

    return CandidateSet(
        name="H_W",
        candidates=candidates,
        threshold_config=threshold_config,
    )


def create_validated_set(
    discovery_set: CandidateSet,
    activation_confidences: Dict[Any, float],
    geometry_confidences: Dict[Any, float],
    threshold_config: Optional[ThresholdConfig] = None,
) -> CandidateSet:
    """
    Create H_cand: Validated candidates passing all domain thresholds.

    H_cand = {p : p ∈ H_W ∧ C_A⁽μ⁽ᵖ⁾⁾ ≥ τ_A ∧ C_G⁽μ⁽ᵖ⁾⁾ ≥ τ_G}

    Args:
        discovery_set: H_W from create_discovery_set
        activation_confidences: {component_id: C_A} mapping
        geometry_confidences: {component_id: C_G} mapping
        threshold_config: Override thresholds (uses discovery_set's if None)

    Returns:
        CandidateSet with validated candidates
    """
    config = threshold_config or discovery_set.threshold_config or ThresholdConfig()
    candidates = {}

    for param_id, info in discovery_set.candidates.items():
        # Get component for this parameter
        component = info.component

        # Look up component-level confidences
        c_a = activation_confidences.get(component) if component else None
        c_g = geometry_confidences.get(component) if component else None

        # If a domain was not run, its confidence map is empty and should not
        # filter. If it was run, missing component evidence is a failed join.
        passes_a = not activation_confidences or (c_a is not None and c_a >= config.tau_A)
        passes_g = not geometry_confidences or (c_g is not None and c_g >= config.tau_G)

        if passes_a and passes_g:
            candidates[param_id] = CandidateInfo(
                id=param_id,
                status=CandidateStatus.VALIDATED,
                C_W=info.C_W,
                C_A=c_a,
                C_G=c_g,
                component=component,
                metadata=info.metadata.copy(),
            )

    return CandidateSet(
        name="H_cand",
        candidates=candidates,
        threshold_config=config,
        parent_set="H_W",
    )


def create_confirmed_set(
    validated_set: CandidateSet,
    ablation_results: Dict[Any, bool],
) -> CandidateSet:
    """
    Create H_conf: Confirmed candidates after ablation testing.

    H_conf = {p ∈ H_cand : ablation affects behavior without breaking model}

    Args:
        validated_set: H_cand from create_validated_set
        ablation_results: {param_id: True if confirmed, False if rejected}

    Returns:
        CandidateSet with confirmed candidates
    """
    candidates = {}

    for param_id, info in validated_set.candidates.items():
        if ablation_results.get(param_id, False):
            new_info = CandidateInfo(
                id=param_id,
                status=CandidateStatus.CONFIRMED,
                C_W=info.C_W,
                C_A=info.C_A,
                C_G=info.C_G,
                C_unified=info.C_unified,
                component=info.component,
                metadata=info.metadata.copy(),
            )
            candidates[param_id] = new_info

    return CandidateSet(
        name="H_conf",
        candidates=candidates,
        threshold_config=validated_set.threshold_config,
        parent_set="H_cand",
    )


def create_multi_path_discovery_set(
    H_W: Set[Any],
    H_A: Set[Any],
    H_G: Set[Any],
    discovery_info: Dict[Any, DiscoveryInfo],
    S_W: Optional[Dict[Any, float]] = None,
    threshold_config: Optional[ThresholdConfig] = None,
    component_map: Optional[Dict[Any, Any]] = None,
) -> CandidateSet:
    """
    Create H_disc: Multi-path discovery set.

    H_disc = H_W ∪ H_A ∪ H_G

    Each parameter discovered by ANY path is included. Parameters discovered
    by multiple paths receive a multi-path bonus in their unified confidence.

    Args:
        H_W: Weight discovery set (param IDs)
        H_A: Activation discovery set (param IDs)
        H_G: Gradient discovery set (param IDs)
        discovery_info: {param_id: DiscoveryInfo} with path info
        S_W: Optional {param_id: weight discovery score} for C_W
        threshold_config: Threshold configuration
        component_map: Optional {param_id: component_id} mapping

    Returns:
        CandidateSet with all discovered candidates
    """
    H_disc = H_W | H_A | H_G
    candidates = {}

    for param_id in H_disc:
        info = discovery_info.get(param_id)
        c_w = S_W.get(param_id) if S_W else None

        candidates[param_id] = CandidateInfo(
            id=param_id,
            status=CandidateStatus.DISCOVERY,
            C_W=c_w,
            discovery=info,
            component=component_map.get(param_id) if component_map else None,
        )

    return CandidateSet(
        name="H_disc",
        candidates=candidates,
        threshold_config=threshold_config or ThresholdConfig(),
    )


__all__ = [
    "CandidateStatus",
    "CandidateInfo",
    "CandidateSet",
    "create_discovery_set",
    "create_multi_path_discovery_set",
    "create_validated_set",
    "create_confirmed_set",
]
