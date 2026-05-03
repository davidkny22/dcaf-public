"""Cross-run consistency analysis.

Analyzes patterns across multiple DCAF training runs to identify
stable core parameters and variable parameters. Helps assess
reproducibility and robustness of circuit identification.

Extracted from training/trainer.py — analysis is a diagnostic concern.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record of a single DCAF run for cross-run comparison."""

    run_id: str
    timestamp: str
    model_name: str
    config: Dict[str, Any]
    candidate_params: List[str]
    validated_params: List[str]
    total_parameters: int = 0

    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any],
               candidates: List[str], validated: List[str],
               total_params: int = 0) -> "RunRecord":
        return cls(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            config=config,
            candidate_params=candidates,
            validated_params=validated,
            total_parameters=total_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "config": self.config,
            "candidate_params": self.candidate_params,
            "validated_params": self.validated_params,
            "total_parameters": self.total_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        return cls(**data)


@dataclass
class CrossRunAnalysis:
    """Results of cross-run pattern analysis."""

    total_runs: int
    consistent_params: Set[str]
    variable_params: Set[str]
    consistency_scores: Dict[str, float]

    @property
    def stable_core_size(self) -> int:
        return len(self.consistent_params)

    @property
    def variable_size(self) -> int:
        return len(self.variable_params)


def save_run_record(record: RunRecord, output_dir: Path) -> Path:
    """Save a run record to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_{record.run_id}.json"
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)
    return path


def load_run_records(records_dir: Path) -> List[RunRecord]:
    """Load all run records from a directory."""
    records_dir = Path(records_dir)
    records = []
    for path in sorted(records_dir.glob("run_*.json")):
        with open(path) as f:
            records.append(RunRecord.from_dict(json.load(f)))
    return records


def analyze_cross_run_patterns(
    records: List[RunRecord],
    min_consistency: float = 0.8,
) -> CrossRunAnalysis:
    """Analyze parameter consistency across multiple runs.

    Args:
        records: List of RunRecord from multiple runs
        min_consistency: Threshold for stable core (default: 80% of runs)

    Returns:
        CrossRunAnalysis with consistent/variable parameter sets
    """
    if not records:
        return CrossRunAnalysis(0, set(), set(), {})

    param_counts: Dict[str, int] = {}
    for record in records:
        for param in record.validated_params:
            param_counts[param] = param_counts.get(param, 0) + 1

    n_runs = len(records)
    scores = {p: count / n_runs for p, count in param_counts.items()}

    consistent = {p for p, score in scores.items() if score >= min_consistency}
    variable = {p for p, score in scores.items() if score < 0.5}

    logger.info(f"Cross-run analysis: {n_runs} runs, "
                f"{len(consistent)} stable core, {len(variable)} variable")

    return CrossRunAnalysis(
        total_runs=n_runs,
        consistent_params=consistent,
        variable_params=variable,
        consistency_scores=scores,
    )


__all__ = [
    "RunRecord",
    "CrossRunAnalysis",
    "save_run_record",
    "load_run_records",
    "analyze_cross_run_patterns",
]
