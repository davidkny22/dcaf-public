"""Composable training signal construction (Def 1.7).

Builds training runs from composable flags. No static variant lists.
Each flag adds signals to the T+, T-, or T0 cluster independently.

Usage:
    runs = build_signal_runs(prefopt=True, sft=True, anti=True)
    # Returns 10 runs: 5 T+ (PrefOpt, SFT, Anti target + Anti-Neg target)
    #                   5 T- (mirrors) + 1 T0 (always included)
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SignalRun:
    """A single training run specification."""

    signal_id: str
    name: str
    cluster: Literal["+", "-", "0"]
    training_type: Literal["SFT", "PrefOpt", "Cumulative", "Anti", "Negated", "DomainNative"]
    direction: Literal["target", "opposite", "neutral"]
    requires_checkpoint: bool = False
    checkpoint_source: str = ""

    def __repr__(self) -> str:
        return f"SignalRun({self.signal_id}, {self.name}, T{self.cluster})"


def build_signal_runs(
    prefopt: bool = True,
    sft: bool = False,
    cumulative: bool = False,
    anti: bool = False,
    negated: bool = False,
    target_only: bool = False,
    opposite_only: bool = False,
) -> List[SignalRun]:
    """Build training runs from composable flags (Def 1.7).

    Args:
        prefopt: Include preference optimization signals (t1/t6). Always True — core signal.
        sft: Include supervised fine-tuning signals (t2/t7).
        cumulative: Include SFT→PrefOpt sequential signals (t3/t8). Requires sft=True.
        anti: Include gradient ascent signals (t4/t9).
        negated: Include unlearning signals (t5/t10). Requires PrefOpt checkpoint.
        target_only: Only T+ signals (skip T-).
        opposite_only: Only T- signals (skip T+).

    Returns:
        List of SignalRun objects. Always includes t11 (DomainNative baseline).
    """
    if cumulative and not sft:
        raise ValueError("Cumulative (t3/t8) requires sft=True (SFT runs first)")

    runs: List[SignalRun] = []
    signal_counter = {"target": 0, "opposite": 0}

    def _add_pair(training_type, target_name, opposite_name,
                  requires_checkpoint=False, checkpoint_source=""):
        nonlocal signal_counter

        if not opposite_only:
            signal_counter["target"] += 1
            tid = f"t{signal_counter['target']}"
            runs.append(SignalRun(
                signal_id=tid, name=target_name, cluster="+",
                training_type=training_type, direction="target",
                requires_checkpoint=requires_checkpoint,
                checkpoint_source=checkpoint_source,
            ))

        if not target_only:
            signal_counter["opposite"] += 1
            oid = f"t{signal_counter['opposite'] + 5}"
            runs.append(SignalRun(
                signal_id=oid, name=opposite_name, cluster="-",
                training_type=training_type, direction="opposite",
                requires_checkpoint=requires_checkpoint,
                checkpoint_source=checkpoint_source,
            ))

    if prefopt:
        _add_pair("PrefOpt", "PrefOpt(target>opposite)", "PrefOpt(opposite>target)")

    if sft:
        _add_pair("SFT", "SFT(target)", "SFT(opposite)")

    if cumulative:
        _add_pair("Cumulative", "Cumulative(target)", "Cumulative(opposite)")

    if anti:
        _add_pair("Anti", "Anti(opposite>target)", "Anti(target>opposite)")

    if negated:
        _add_pair("Negated", "Negated-(opposite>target)", "Negated+(target>opposite)",
                  requires_checkpoint=True, checkpoint_source="prefopt")

    runs.append(SignalRun(
        signal_id="t11", name="DomainNative(neutral)", cluster="0",
        training_type="DomainNative", direction="neutral",
    ))

    return runs


def get_target_runs(runs: List[SignalRun]) -> List[SignalRun]:
    """Filter to T+ runs only."""
    return [r for r in runs if r.cluster == "+"]


def get_opposite_runs(runs: List[SignalRun]) -> List[SignalRun]:
    """Filter to T- runs only."""
    return [r for r in runs if r.cluster == "-"]


def get_baseline_runs(runs: List[SignalRun]) -> List[SignalRun]:
    """Filter to T0 runs only."""
    return [r for r in runs if r.cluster == "0"]


__all__ = [
    "SignalRun",
    "build_signal_runs",
    "get_target_runs",
    "get_opposite_runs",
    "get_baseline_runs",
]
