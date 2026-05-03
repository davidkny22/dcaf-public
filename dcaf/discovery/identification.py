"""Circuit candidate identification via weight delta magnitude (§3, Def 2.1).

Identifies projections with significant weight deltas across training signals
using percentile-based thresholding. This is the initial candidate screening
step that feeds into multi-path discovery.

Extracted from training/trainer.py — this logic belongs in the discovery layer,
not the training layer.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from dcaf.arch.transformer import parse_param_metadata, should_exclude_param
from dcaf.core.defaults import TAU_SIG, TOP_K_CANDIDATES

logger = logging.getLogger(__name__)


def identify_candidates(
    signal_deltas: Dict[str, Dict[str, torch.Tensor]],
    base_weights: Dict[str, torch.Tensor],
    threshold: Optional[float] = None,
    percentile: float = TAU_SIG,
    top_k: int = TOP_K_CANDIDATES,
    exclude_general: bool = True,
) -> List[Dict]:
    """Find projections with significant weight deltas across signals.

    A projection is a candidate if its maximum delta magnitude across
    all signals exceeds the percentile-based threshold.

    Args:
        signal_deltas: {signal_id: {param_name: delta_tensor}}
        base_weights: {param_name: base_weight_tensor} for relative magnitude
        threshold: Absolute threshold (if None, computed from percentile)
        percentile: Percentile for threshold (default: TAU_SIG = 85)
        top_k: Maximum candidates to return
        exclude_general: Whether to exclude embeddings, layer norms, etc.

    Returns:
        List of candidate dicts sorted by magnitude, each with:
        - parameter_name, signal_id, delta_magnitude, relative_magnitude,
          layer, component_type
    """
    all_params = set()
    for deltas in signal_deltas.values():
        all_params.update(deltas.keys())

    all_magnitudes = []
    param_max = {}

    for param_name in all_params:
        signal_magnitudes = {}
        for signal_id, deltas in signal_deltas.items():
            if param_name in deltas:
                mag = torch.norm(deltas[param_name]).item()
                signal_magnitudes[signal_id] = mag
                all_magnitudes.append(mag)

        if signal_magnitudes:
            max_signal = max(signal_magnitudes, key=signal_magnitudes.get)
            param_max[param_name] = (signal_magnitudes[max_signal], max_signal)

    if not all_magnitudes:
        logger.warning("No deltas found — cannot identify candidates")
        return []

    magnitudes_np = np.array(all_magnitudes)
    if magnitudes_np.max() <= 0:
        logger.warning("All deltas are zero — no candidates to identify")
        return []
    if threshold is None:
        threshold = float(np.percentile(magnitudes_np, percentile))
        if threshold <= 0:
            threshold = float(magnitudes_np.max()) * 0.01

    logger.info(f"Delta stats: mean={magnitudes_np.mean():.6f}, "
                f"median={np.median(magnitudes_np):.6f}, "
                f"threshold ({percentile}th pctl)={threshold:.6f}")

    candidates = []
    for param_name, (max_mag, max_signal) in param_max.items():
        if max_mag < threshold:
            continue

        if exclude_general and should_exclude_param(param_name):
            continue

        original_norm = torch.norm(base_weights.get(param_name, torch.tensor(0.0))).item()
        relative_mag = max_mag / (original_norm + 1e-8)

        meta = parse_param_metadata(param_name)

        candidates.append({
            "parameter_name": param_name,
            "signal_id": max_signal,
            "delta_magnitude": max_mag,
            "relative_magnitude": relative_mag,
            "layer": meta["layer"],
            "component_type": meta["component"],
        })

    candidates.sort(key=lambda c: c["delta_magnitude"], reverse=True)
    candidates = candidates[:top_k]

    logger.info(f"Identified {len(candidates)} candidate projections")
    return candidates


__all__ = ["identify_candidates"]
