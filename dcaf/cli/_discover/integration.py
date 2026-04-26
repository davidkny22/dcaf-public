"""
Discovery integration and output (§3 Multi-Path Discovery: H_disc = H_W ∪ H_A ∪ H_G).

Combines discovery sets from all three paths into a unified discovery result
and serialises it to JSON for downstream analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Any, List, Optional

from dcaf.core.defaults import BETA_PATH
from dcaf.discovery import (
    compute_discovery_union,
    compute_all_discovery_info,
    create_discovery_result,
    DiscoveryResult,
)

logger = logging.getLogger(__name__)


def create_discovery_output(
    H_W: Set[Any],
    H_A: Set[Any],
    H_G: Set[Any],
    S_W: Dict[Any, float],
    S_A: Dict[Any, float],
    S_G: Dict[Any, float],
    run_path: Path,
    config: Dict[str, Any],
    beta_path: float = BETA_PATH,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create discovery output for JSON serialization.

    Args:
        H_W: Weight discovery set
        H_A: Activation discovery set
        H_G: Gradient discovery set
        S_W: Weight discovery scores
        S_A: Activation discovery scores
        S_G: Gradient discovery scores
        run_path: Path to run directory
        config: Discovery configuration
        beta_path: Multi-path bonus weight

    Returns:
        Dict ready for JSON serialization
    """
    # Compute union and discovery info
    result = create_discovery_result(
        H_W=H_W,
        H_A=H_A,
        H_G=H_G,
        S_W=S_W,
        S_A=S_A,
        S_G=S_G,
        beta_path=beta_path,
    )

    # Build output dict
    output = {
        "version": "1.0",
        "run_path": str(run_path),
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "discovery_sets": {
            "H_W": sorted([serialize_key(k) for k in H_W]),
            "H_A": sorted([serialize_key(k) for k in H_A]),
            "H_G": sorted([serialize_key(k) for k in H_G]),
        },
        "H_disc": sorted([serialize_key(k) for k in result.H_disc]),
        "discovery_info": {
            serialize_key(k): info.to_dict()
            for k, info in result.discovery_info.items()
        },
        "scores": {
            "S_W": {serialize_key(k): v for k, v in S_W.items()},
            "S_A": {serialize_key(k): v for k, v in S_A.items()},
            "S_G": {serialize_key(k): v for k, v in S_G.items()},
        },
        "summary": result.summary(),
    }

    # Add param index-to-name mapping for human-readable inspection
    if param_names:
        output["param_index_to_name"] = {
            str(i): name for i, name in enumerate(param_names)
        }

    return output


def serialize_key(key: Any) -> str:
    """
    Serialize a discovery set key for JSON.

    Handles int indices and string param names.
    """
    if isinstance(key, int):
        return str(key)
    return key


def save_discovery_result(result: Dict[str, Any], output_path: Path) -> None:
    """
    Save discovery result to JSON file.

    Args:
        result: Discovery output dict
        output_path: Path to save to
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Discovery results saved to: {output_path}")


def load_discovery_result(input_path: Path) -> Dict[str, Any]:
    """
    Load discovery result from JSON file.

    Args:
        input_path: Path to load from

    Returns:
        Discovery output dict
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discovery_result_to_sets(
    result: Dict[str, Any],
) -> tuple:
    """
    Convert loaded discovery result back to sets.

    Args:
        result: Loaded JSON dict

    Returns:
        (H_disc, discovery_info, H_W, H_A, H_G)
    """
    from dcaf.discovery import DiscoveryInfo

    # Convert H_disc
    H_disc = set()
    for k in result.get("H_disc", []):
        # Try to convert to int if it looks like an index
        try:
            H_disc.add(int(k))
        except ValueError:
            H_disc.add(k)

    # Convert discovery sets
    H_W = set()
    for k in result.get("discovery_sets", {}).get("H_W", []):
        try:
            H_W.add(int(k))
        except ValueError:
            H_W.add(k)

    H_A = set()
    for k in result.get("discovery_sets", {}).get("H_A", []):
        try:
            H_A.add(int(k))
        except ValueError:
            H_A.add(k)

    H_G = set()
    for k in result.get("discovery_sets", {}).get("H_G", []):
        try:
            H_G.add(int(k))
        except ValueError:
            H_G.add(k)

    # Convert discovery info
    discovery_info = {}
    for k, info_dict in result.get("discovery_info", {}).items():
        try:
            key = int(k)
        except ValueError:
            key = k

        discovery_info[key] = DiscoveryInfo.from_dict(info_dict)

    return H_disc, discovery_info, H_W, H_A, H_G


__all__ = [
    "create_discovery_output",
    "save_discovery_result",
    "load_discovery_result",
    "discovery_result_to_sets",
]
