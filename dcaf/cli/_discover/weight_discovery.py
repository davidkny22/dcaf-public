"""
Weight-based discovery runner (H_W) (§3 Multi-Path Discovery, §4 Weight Analysis).

Identifies named parameters that changed significantly during behavioral training.
Operates at named-parameter granularity (one weight matrix = one unit).

Algorithm:
  - Significance: torch.norm(delta[param]) >= percentile_threshold
  - Opposition: T+ and T- deltas have opposing mean signs
  - Baseline filter: NOT significant in T0 (language/neutral) signals
"""

import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import torch
from torch import Tensor

from dcaf.arch.transformer import should_exclude_param
from dcaf.core.defaults import TAU_SIG, TAU_BASE
from dcaf.core.signals import CANONICAL_SIGNALS
from dcaf.storage.delta_store import DeltaStore

logger = logging.getLogger(__name__)


def _compute_param_norms(
    delta: Dict[str, Tensor],
) -> Dict[str, float]:
    """Compute L2 norm for each named parameter in a delta."""
    return {name: torch.norm(tensor).item() for name, tensor in delta.items()}


def _percentile_threshold(norms: Dict[str, float], percentile: float) -> float:
    """Compute percentile threshold over parameter norms."""
    if not norms:
        return 0.0
    values = list(norms.values())
    return float(np.percentile(values, percentile))


def _is_significant(
    param_name: str,
    delta: Dict[str, Tensor],
    threshold: float,
) -> bool:
    """Check if a named parameter's delta norm exceeds the threshold."""
    if param_name not in delta:
        return False
    return torch.norm(delta[param_name]).item() >= threshold


def _sign_opposes(
    param_name: str,
    delta_a: Dict[str, Tensor],
    delta_b: Dict[str, Tensor],
) -> bool:
    """Check if two deltas push a parameter in opposing directions."""
    tensor_a = delta_a.get(param_name)
    tensor_b = delta_b.get(param_name)
    if tensor_a is None or tensor_b is None:
        return False
    sign_a = 1 if tensor_a.mean().item() >= 0 else -1
    sign_b = 1 if tensor_b.mean().item() >= 0 else -1
    return sign_a != sign_b


def run_weight_discovery(
    run_path: Path,
    tau_sig: float = TAU_SIG,
    tau_base: float = TAU_BASE,
) -> Tuple[Set[str], Dict[str, float], List[str]]:
    """
    Run weight-based discovery at named-parameter granularity.

    For each named parameter (e.g. "gpt_neox.layers.5.mlp.dense_h_to_4h"):
    1. Compute norm of its delta in each signal
    2. Check significance against percentile threshold
    3. Check opposition between T+ and T- signals
    4. Filter out parameters significant in T0 (language baseline)

    Score: S_W = (fraction of signals where significant) with opposition bonus.
    H_W = parameters with S_W > 0.

    Args:
        run_path: Path to DCAF run directory
        tau_sig: Significance percentile threshold (default 85)
        tau_base: Baseline NOT-significant threshold (default 50)

    Returns:
        (H_W set of param names, {param_name: S_W score}, all_param_names list)
    """
    # Load deltas
    delta_store = DeltaStore(run_path)
    available_deltas = delta_store.list_deltas()

    if not available_deltas:
        logger.error(f"No deltas found in {run_path}")
        return set(), {}, []

    logger.info(f"Found {len(available_deltas)} deltas:")
    for name in available_deltas:
        logger.info(f"  - {name}")

    # Load all deltas as Dict[signal_name, Dict[param_name, Tensor]]
    logger.info("\nLoading deltas...")
    deltas: Dict[str, Dict[str, Tensor]] = {}
    for name in available_deltas:
        deltas[name] = delta_store.load_delta(name)

    # Collect all parameter names
    all_params: Set[str] = set()
    for delta in deltas.values():
        all_params.update(delta.keys())
    param_names = sorted(all_params)

    logger.info(f"Total named parameters: {len(param_names)}")

    # Detect signal clusters present in this run.
    # New delta names follow the pattern delta_t{N}_{name}, e.g. delta_t1_prefopt_target.
    # Cluster membership is determined by CANONICAL_SIGNALS: cluster "+" = T+, "-" = T-, "0" = T0.
    target_ids = {f"delta_{s.id}" for s in CANONICAL_SIGNALS if s.cluster == "+"}
    opposite_ids = {f"delta_{s.id}" for s in CANONICAL_SIGNALS if s.cluster == "-"}
    baseline_ids = {f"delta_{s.id}" for s in CANONICAL_SIGNALS if s.cluster == "0"}

    # Also accept legacy delta names that embed the run_type token
    def _classify(name: str) -> str:
        """Return '+', '-', or '0' for a delta name."""
        # Exact match to canonical delta names (delta_t1_prefopt_target, etc.)
        for sig in CANONICAL_SIGNALS:
            if name == f"delta_{sig.id}" or name.startswith(f"delta_{sig.id}_"):
                return sig.cluster
        # Heuristic fallback for legacy names
        n = name.lower()
        if "language" in n or "neutral" in n or "baseline" in n:
            return "0"
        if "target" in n or ("safe" in n and "anti" not in n and "negated" not in n):
            return "+"
        if "opposite" in n or ("adv" in n and "anti" not in n):
            return "-"
        return "+"  # default: treat as behavioral T+

    t_plus_present = {n: deltas[n] for n in deltas if _classify(n) == "+"}
    t_minus_present = {n: deltas[n] for n in deltas if _classify(n) == "-"}
    t_zero_present = {n: deltas[n] for n in deltas if _classify(n) == "0"}

    logger.info("\nSignal clusters detected:")
    logger.info(f"  T+ (target): {len(t_plus_present)} - {list(t_plus_present.keys())}")
    logger.info(f"  T- (opposite): {len(t_minus_present)} - {list(t_minus_present.keys())}")
    logger.info(f"  T0 (baseline): {len(t_zero_present)} - {list(t_zero_present.keys())}")

    # Precompute thresholds (once per signal)
    sig_thresholds: Dict[str, float] = {}
    base_thresholds: Dict[str, float] = {}

    for signal_name, delta in deltas.items():
        norms = _compute_param_norms(delta)
        sig_thresholds[signal_name] = _percentile_threshold(norms, tau_sig)
        if _classify(signal_name) == "0":
            base_thresholds[signal_name] = _percentile_threshold(norms, tau_base)

    logger.info(f"\nSignificance thresholds (p{tau_sig}):")
    for name, thresh in sig_thresholds.items():
        logger.info(f"  {name}: {thresh:.6f}")
    if base_thresholds:
        logger.info(f"Baseline thresholds (p{tau_base}):")
        for name, thresh in base_thresholds.items():
            logger.info(f"  {name}: {thresh:.6f}")

    # Score each named parameter
    logger.info(f"\nScoring {len(param_names)} parameters...")
    S_W: Dict[str, float] = {}
    H_W: Set[str] = set()

    behavioral_signals = {**t_plus_present, **t_minus_present}
    n_behavioral = len(behavioral_signals) if behavioral_signals else 1

    for param_name in param_names:
        # Skip excluded patterns (embeddings, layernorms, etc.)
        if should_exclude_param(param_name):
            continue

        # 1. Baseline filter: must be INsignificant in ALL T0 signals
        baseline_passed = True
        for t0_name, t0_delta in t_zero_present.items():
            if param_name in t0_delta:
                threshold = base_thresholds[t0_name]
                if threshold == 0.0:
                    continue  # Zero threshold means signal had no changes — skip
                norm = torch.norm(t0_delta[param_name]).item()
                if norm >= threshold:
                    baseline_passed = False
                    break

        if not baseline_passed:
            continue

        # 2. Signal presence: fraction of behavioral signals where significant
        sig_count = 0
        for sig_name, sig_delta in behavioral_signals.items():
            if _is_significant(param_name, sig_delta, sig_thresholds[sig_name]):
                sig_count += 1

        if sig_count == 0:
            continue

        signal_presence = sig_count / n_behavioral

        # 3. Opposition bonus: check if T+ and T- oppose
        has_opposition = False
        for tp_name, tp_delta in t_plus_present.items():
            for tm_name, tm_delta in t_minus_present.items():
                if _sign_opposes(param_name, tp_delta, tm_delta):
                    has_opposition = True
                    break
            if has_opposition:
                break

        opp_bonus = 0.3 if has_opposition else 0.0

        # S_W = min(1, signal_presence + opp_bonus)
        score = min(1.0, signal_presence + opp_bonus)
        S_W[param_name] = score
        H_W.add(param_name)

    # Log summary
    logger.info(f"\nH_W: {len(H_W)} parameters discovered")
    if S_W:
        scores = list(S_W.values())
        logger.info(f"  Mean S_W: {sum(scores) / len(scores):.4f}")
        logger.info(f"  Max S_W: {max(scores):.4f}")

        # Breakdown by component type
        mlp_count = sum(1 for p in H_W if "mlp" in p.lower())
        attn_count = sum(1 for p in H_W if "attention" in p.lower() or "attn" in p.lower())
        other_count = len(H_W) - mlp_count - attn_count
        logger.info(f"  MLP: {mlp_count}, Attention: {attn_count}, Other: {other_count}")

    return H_W, S_W, param_names


__all__ = ["run_weight_discovery"]
