"""
Discovery integration (sec:discovery-integration).

Combines all three discovery paths into the unified discovery set H_disc
and computes DiscoveryInfo (path membership, path count, multi-path bonus).

Unified discovery set (def:unified-discovery-set):
  H_disc = H_W ∪ H_A ∪ H_G

Component discovery attribution (def:component-discovery-attribution):
  disc_paths(k) = paths that include k
  paths(k) = |disc_paths(k)|
  bonus = beta_path * max(0, paths(k) - 1)

A parameter discovered by ANY path receives confidence scores from ALL domains.
Path count drives the multi-path discovery bonus in def:multi-path-discovery-bonus.
"""

from dataclasses import dataclass
from typing import Any, Dict, Set

from dcaf.core.defaults import BETA_PATH
from dcaf.discovery.info import DiscoveryInfo, compute_multi_path_bonus


def compute_discovery_union(
    H_W: Set[Any],
    H_A: Set[Any],
    H_G: Set[Any],
) -> Set[Any]:
    """
    Compute unified discovery set.

    H_disc = H_W ∪ H_A ∪ H_G

    Args:
        H_W: Weight discovery set
        H_A: Activation discovery set
        H_G: Gradient discovery set

    Returns:
        Union of all discovery sets
    """
    return H_W | H_A | H_G


def compute_all_discovery_info(
    H_W: Set[Any],
    H_A: Set[Any],
    H_G: Set[Any],
    S_W: Dict[Any, float],
    S_A: Dict[Any, float],
    S_G: Dict[Any, float],
    beta_path: float = BETA_PATH,
) -> Dict[Any, DiscoveryInfo]:
    """
    Compute DiscoveryInfo for all discovered parameters.

    Args:
        H_W: Weight discovery set
        H_A: Activation discovery set
        H_G: Gradient discovery set
        S_W: {param: weight discovery score}
        S_A: {param: activation discovery score}
        S_G: {param: gradient discovery score}
        beta_path: Multi-path bonus weight

    Returns:
        {param: DiscoveryInfo}
    """
    H_disc = compute_discovery_union(H_W, H_A, H_G)

    discovery_info = {}

    for param in H_disc:
        paths = set()

        if param in H_W:
            paths.add('W')
        if param in H_A:
            paths.add('A')
        if param in H_G:
            paths.add('G')

        path_count = len(paths)
        bonus = compute_multi_path_bonus(path_count, beta_path)

        discovery_info[param] = DiscoveryInfo(
            paths=paths,
            path_count=path_count,
            bonus=bonus,
            S_W=S_W.get(param, 0.0),
            S_A=S_A.get(param, 0.0),
            S_G=S_G.get(param, 0.0),
        )

    return discovery_info


@dataclass
class DiscoveryResult:
    """
    Complete discovery result from all paths.

    Attributes:
        H_disc: Unified discovery set (union)
        H_W: Weight discovery set
        H_A: Activation discovery set
        H_G: Gradient discovery set
        discovery_info: {param: DiscoveryInfo}
        S_W: {param: weight discovery score}
        S_A: {param: activation discovery score}
        S_G: {param: gradient discovery score}
    """
    H_disc: Set[Any]
    H_W: Set[Any]
    H_A: Set[Any]
    H_G: Set[Any]
    discovery_info: Dict[Any, DiscoveryInfo]
    S_W: Dict[Any, float]
    S_A: Dict[Any, float]
    S_G: Dict[Any, float]

    @property
    def total_discovered(self) -> int:
        """Total parameters discovered by any path."""
        return len(self.H_disc)

    @property
    def multi_path_count(self) -> int:
        """Count of parameters discovered by multiple paths."""
        return sum(
            1 for info in self.discovery_info.values()
            if info.is_multi_path
        )

    def get_by_path(self, path: str) -> Set[Any]:
        """Get parameters discovered by a specific path."""
        if path == 'W':
            return self.H_W
        elif path == 'A':
            return self.H_A
        elif path == 'G':
            return self.H_G
        else:
            raise ValueError(f"Unknown path: {path}")

    def get_exclusive_to_path(self, path: str) -> Set[Any]:
        """Get parameters discovered ONLY by this path."""
        path_set = self.get_by_path(path)
        return {
            p for p in path_set
            if self.discovery_info[p].path_count == 1
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        path_exclusive = {
            'W_only': len(self.get_exclusive_to_path('W')),
            'A_only': len(self.get_exclusive_to_path('A')),
            'G_only': len(self.get_exclusive_to_path('G')),
        }

        path_intersections = {
            'W_and_A': len(self.H_W & self.H_A),
            'W_and_G': len(self.H_W & self.H_G),
            'A_and_G': len(self.H_A & self.H_G),
            'all_three': len(self.H_W & self.H_A & self.H_G),
        }

        return {
            'total_discovered': self.total_discovered,
            'by_path': {
                'H_W': len(self.H_W),
                'H_A': len(self.H_A),
                'H_G': len(self.H_G),
            },
            'exclusive': path_exclusive,
            'intersections': path_intersections,
            'multi_path_count': self.multi_path_count,
        }


def create_discovery_result(
    H_W: Set[Any],
    H_A: Set[Any],
    H_G: Set[Any],
    S_W: Dict[Any, float],
    S_A: Dict[Any, float],
    S_G: Dict[Any, float],
    beta_path: float = BETA_PATH,
) -> DiscoveryResult:
    """
    Create complete discovery result.

    Args:
        H_W: Weight discovery set
        H_A: Activation discovery set
        H_G: Gradient discovery set
        S_W: Weight discovery scores
        S_A: Activation discovery scores
        S_G: Gradient discovery scores
        beta_path: Multi-path bonus weight

    Returns:
        DiscoveryResult with all computed values
    """
    H_disc = compute_discovery_union(H_W, H_A, H_G)

    discovery_info = compute_all_discovery_info(
        H_W, H_A, H_G, S_W, S_A, S_G, beta_path
    )

    return DiscoveryResult(
        H_disc=H_disc,
        H_W=H_W,
        H_A=H_A,
        H_G=H_G,
        discovery_info=discovery_info,
        S_W=S_W,
        S_A=S_A,
        S_G=S_G,
    )


__all__ = [
    "compute_discovery_union",
    "compute_all_discovery_info",
    "DiscoveryResult",
    "create_discovery_result",
]
