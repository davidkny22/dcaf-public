"""
Criteria Engine: Evaluates DCAF criteria expressions on loaded deltas.

Supports sec:multi-path-discovery by labeling which signals a parameter responds
to (significant change, sign opposition, etc.). Domain confidence (C_W, C_A, C_G +
triangulation) is the primary analysis system; this engine supplements with signal
labels.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from dcaf.confidence.signals import SignalDetails
from dcaf.core.defaults import LANGUAGE_PERCENTILE, TAU_SIG

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter exclusion patterns - single source of truth in arch.transformer
# =============================================================================

from dcaf.arch.transformer import EXCLUDED_PARAM_PATTERNS, should_exclude_param

# =============================================================================
# Analysis Result
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of running criteria analysis on a DCAF run."""

    run_id: str
    criteria_name: str
    criteria_expr: str
    percentile: float
    language_percentile: float
    matching_params: List[str]
    param_details: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    signal_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Maps param_name to signal tracking details:
    {
        "signals": {"delta_t1_prefopt_target": True, ...},
        "signal_count": int,
        "has_opposition": bool,
        "relevance_confidence": float  # 0.0-1.0
    }
    """

    @property
    def matching_count(self) -> int:
        return len(self.matching_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        # Ensure param_details include ablation_validated, safety_score, and signal details
        enriched_params = []
        for p in self.param_details:
            param_name = p.get("name")

            # Get signal details for this param if available
            sig_detail = self.signal_details.get(param_name, {})

            param_entry = {
                "name": param_name,
                "layer": p.get("layer"),
                "component": p.get("component"),
                "ablation_validated": p.get("ablation_validated", False),
                "safety_score": p.get("safety_score"),
                "sae_correlated": p.get("sae_correlated"),
                "confidence": p.get("confidence", 1.0),
                # Add signal tracking fields if available
                "signals": sig_detail.get("signals", {}),
                "signal_count": sig_detail.get("signal_count", 0),
                "has_opposition": sig_detail.get("has_opposition", False),
                "relevance_confidence": sig_detail.get("relevance_confidence", 0.0),
            }
            enriched_params.append(param_entry)

        return {
            "run_id": self.run_id,
            "criteria_name": self.criteria_name,
            "criteria_expr": self.criteria_expr,
            "percentile": self.percentile,
            "language_percentile": self.language_percentile,
            "matching_count": self.matching_count,
            "thresholds": self.thresholds,
            "summary": self.summary,
            "matching_params": enriched_params,
            "signal_details": self.signal_details,
        }


# =============================================================================
# Criteria Engine
# =============================================================================

class ParamCriteriaEngine:
    """
    Evaluates DCAF criteria expressions on loaded delta tensors.

    Works independently of training - loads deltas from disk and evaluates
    criteria expressions to find matching parameters.
    """

    def __init__(
        self,
        deltas: Dict[str, Dict[str, torch.Tensor]],
        percentile: float = TAU_SIG,
        language_percentile: float = LANGUAGE_PERCENTILE,
    ):
        """
        Initialize ParamCriteriaEngine with loaded deltas.

        Args:
            deltas: {delta_name: {param_name: tensor}} mapping
            percentile: Percentile for "significant" threshold (default TAU_SIG=85)
            language_percentile: Percentile for "NOT significant" (default LANGUAGE_PERCENTILE=50)
        """
        self.deltas = deltas
        self.percentile = percentile
        self.language_percentile = language_percentile
        self._threshold_cache: Dict[str, float] = {}

    @classmethod
    def from_delta_store(
        cls,
        delta_store,
        percentile: float = TAU_SIG,
        language_percentile: float = LANGUAGE_PERCENTILE,
    ) -> "ParamCriteriaEngine":
        """
        Create ParamCriteriaEngine by loading all available deltas from DeltaStore.

        Args:
            delta_store: DeltaStore instance
            percentile: Significance threshold percentile
            language_percentile: NOT-significant threshold percentile

        Returns:
            ParamCriteriaEngine with all available deltas loaded
        """
        available = delta_store.list_deltas()
        deltas = {}
        for delta_name in available:
            deltas[delta_name] = delta_store.load_delta(delta_name)

        return cls(deltas, percentile, language_percentile)

    # =========================================================================
    # Threshold computation
    # =========================================================================

    def compute_threshold(self, delta_name: str, percentile: float) -> float:
        """
        Compute significance threshold for a delta using percentile.

        Args:
            delta_name: Name of the delta
            percentile: Percentile for threshold

        Returns:
            Threshold value
        """
        cache_key = f"{delta_name}:{percentile}"
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        if delta_name not in self.deltas:
            raise KeyError(f"Delta '{delta_name}' not loaded")

        delta = self.deltas[delta_name]
        magnitudes = [torch.norm(d).item() for d in delta.values()]
        threshold = float(np.percentile(magnitudes, percentile))

        self._threshold_cache[cache_key] = threshold
        return threshold

    def get_threshold(self, delta_name: str, for_not_significant: bool = False) -> float:
        """Get threshold for significant/not-significant checks."""
        pct = self.language_percentile if for_not_significant else self.percentile
        return self.compute_threshold(delta_name, pct)

    # =========================================================================
    # Predicates
    # =========================================================================

    def significant(self, param_name: str, delta_name: str) -> bool:
        """Check if parameter has significant change in delta."""
        if delta_name not in self.deltas:
            return False
        delta = self.deltas[delta_name]
        if param_name not in delta:
            return False

        magnitude = torch.norm(delta[param_name]).item()
        threshold = self.get_threshold(delta_name, for_not_significant=False)
        return magnitude >= threshold

    def not_significant(self, param_name: str, delta_name: str) -> bool:
        """Check if parameter has insignificant change in delta."""
        if delta_name not in self.deltas:
            return True  # Missing delta = no change
        delta = self.deltas[delta_name]
        if param_name not in delta:
            return True  # Missing param = no change

        magnitude = torch.norm(delta[param_name]).item()
        threshold = self.get_threshold(delta_name, for_not_significant=True)
        return magnitude < threshold

    def sign_opposes(self, param_name: str, delta_a: str, delta_b: str) -> bool:
        """Check if two deltas have opposing signs for this parameter."""
        if delta_a not in self.deltas or delta_b not in self.deltas:
            return False

        delta_a_tensor = self.deltas[delta_a].get(param_name)
        delta_b_tensor = self.deltas[delta_b].get(param_name)

        if delta_a_tensor is None or delta_b_tensor is None:
            return False

        sign_a = 1 if delta_a_tensor.mean().item() >= 0 else -1
        sign_b = 1 if delta_b_tensor.mean().item() >= 0 else -1
        return sign_a != sign_b

    def sign_agrees(self, param_name: str, delta_a: str, delta_b: str) -> bool:
        """Check if two deltas move in the same direction for this parameter."""
        if delta_a not in self.deltas or delta_b not in self.deltas:
            return False

        delta_a_tensor = self.deltas[delta_a].get(param_name)
        delta_b_tensor = self.deltas[delta_b].get(param_name)

        if delta_a_tensor is None or delta_b_tensor is None:
            return False

        sign_a = 1 if delta_a_tensor.mean().item() >= 0 else -1
        sign_b = 1 if delta_b_tensor.mean().item() >= 0 else -1
        return sign_a == sign_b

    # =========================================================================
    # Expression evaluation
    # =========================================================================

    def evaluate(self, param_name: str, criteria_expr: str) -> bool:
        """
        Evaluate intersection criteria expression for a parameter.

        Parses and evaluates expressions like:
        "significant(delta_safe_simpo) AND NOT significant(delta_language)"

        Note: This is a backward compatibility wrapper. For signal tracking,
        use evaluate_with_signals() instead.
        """
        result, _ = self.evaluate_with_signals(param_name, criteria_expr)
        return result

    def evaluate_with_signals(
        self, param_name: str, criteria_expr: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate criteria expression and return passing signals.

        Returns:
            (matches, signal_info) where signal_info contains:
            {
                "signals": Set[str],  # Delta names that passed
                "has_opposition": bool  # Whether any opposition predicate passed
            }
        """
        # Simple expression parser
        expr = criteria_expr.replace(" AND ", " & ").replace(" OR ", " | ")

        # Accumulate signals and track opposition
        result = True
        all_signals = set()
        has_opposition = False

        for clause in expr.split(" & "):
            clause = clause.strip()
            clause_result, clause_signals = self._evaluate_clause(param_name, clause)

            result = result and clause_result
            if not result:
                break

            # Accumulate signals from passing clauses
            if clause_result and clause_signals:
                all_signals.update(clause_signals)

                # Check if this clause was an opposition predicate
                if "opposes" in clause or "agrees" in clause:
                    has_opposition = True

        signal_info = {
            "signals": all_signals,
            "has_opposition": has_opposition,
        }

        return result, signal_info

    def _evaluate_clause(self, param_name: str, clause: str) -> Tuple[bool, Set[str]]:
        """
        Evaluate a single clause and return passing signals.

        Returns:
            (result, passing_signals) where passing_signals are delta names that
            contributed to a TRUE result
        """
        clause = clause.strip()

        # Handle NOT prefix - exclusions don't contribute signals
        if clause.startswith("NOT "):
            inner = clause[4:].strip()
            result, _ = self._evaluate_clause(param_name, inner)
            return (not result, set())

        # Parse function call
        if clause.startswith("significant(") and clause.endswith(")"):
            delta_name = clause[12:-1]
            result = self.significant(param_name, delta_name)
            return (result, {delta_name} if result else set())

        if clause.startswith("sign_opposes(") and clause.endswith(")"):
            args = clause[13:-1].split(",")
            if len(args) == 2:
                delta_a, delta_b = args[0].strip(), args[1].strip()
                result = self.sign_opposes(param_name, delta_a, delta_b)
                # Opposition predicates contribute BOTH deltas if true
                return (result, {delta_a, delta_b} if result else set())

        if clause.startswith("agrees(") and clause.endswith(")"):
            args = clause[7:-1].split(",")
            if len(args) == 2:
                delta_a, delta_b = args[0].strip(), args[1].strip()
                result = self.sign_agrees(param_name, delta_a, delta_b)
                # Agreement predicates contribute BOTH deltas if true
                return (result, {delta_a, delta_b} if result else set())

        logger.warning(f"Unknown clause: {clause}")
        return (False, set())

    # =========================================================================
    # Matching
    # =========================================================================

    def find_matching_params(
        self,
        criteria_expr: str,
        exclude_patterns: bool = True,
    ) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
        """
        Find all parameters that match the intersection criteria.

        Args:
            criteria_expr: Intersection criteria expression
            exclude_patterns: Whether to exclude embed/norm/etc patterns

        Returns:
            (matching_params, signal_details) where signal_details maps
            param_name to signal tracking info
        """
        # Collect all parameter names from all loaded deltas
        all_params = set()
        for delta in self.deltas.values():
            all_params.update(delta.keys())

        matching = set()
        signal_details = {}

        for param_name in all_params:
            if exclude_patterns and should_exclude_param(param_name):
                continue

            # Use new evaluate_with_signals method
            result, signal_info = self.evaluate_with_signals(param_name, criteria_expr)
            if result:
                matching.add(param_name)
                # Store signal info for this parameter
                signal_details[param_name] = signal_info

        return matching, signal_details

    # =========================================================================
    # Analysis
    # =========================================================================

    def analyze(
        self,
        run_id: str,
        criteria_expr: str,
        criteria_name: str = "custom",
    ) -> AnalysisResult:
        """
        Run full labeling analysis for a criteria expression.

        Args:
            run_id: ID of the DCAF run
            criteria_expr: Criteria expression to evaluate
            criteria_name: Label for this analysis

        Returns:
            AnalysisResult with matching params and signal details
        """

        # Get matching params and their signal details
        matching, raw_signal_details = self.find_matching_params(criteria_expr)

        # Compute thresholds for all loaded deltas
        thresholds = {}
        for delta_name in self.deltas.keys():
            thresholds[delta_name] = self.get_threshold(delta_name)

        # Compute confidence scores for each parameter
        total_available_signals = len(self.deltas)
        enriched_signal_details = {}

        for param_name, signal_info in raw_signal_details.items():
            # Create SignalDetails with confidence computation
            signal_detail = SignalDetails.from_signals(
                passing_signals=signal_info["signals"],
                total_available=total_available_signals,
                has_opposition=signal_info["has_opposition"],
            )
            enriched_signal_details[param_name] = signal_detail.to_dict()

        # Build parameter details
        param_details = []
        for param_name in sorted(matching):
            detail = self._build_param_detail(param_name)
            param_details.append(detail)

        # Build summary
        summary = self._build_summary(matching)

        return AnalysisResult(
            run_id=run_id,
            criteria_name=criteria_name,
            criteria_expr=criteria_expr,
            percentile=self.percentile,
            language_percentile=self.language_percentile,
            matching_params=sorted(matching),
            param_details=param_details,
            thresholds=thresholds,
            summary=summary,
            signal_details=enriched_signal_details,
        )

    def _build_param_detail(self, param_name: str) -> Dict[str, Any]:
        """Build detail dict for a parameter."""
        detail = {"name": param_name}

        # Extract layer number if present
        layer_match = re.search(r"layers?\.(\d+)", param_name)
        if layer_match:
            detail["layer"] = int(layer_match.group(1))

        # Extract component type
        if "mlp" in param_name.lower():
            detail["component"] = "mlp"
        elif "attention" in param_name.lower() or "attn" in param_name.lower():
            detail["component"] = "attention"
        else:
            detail["component"] = "other"

        # Add magnitudes from each delta
        magnitudes = {}
        for delta_name, delta in self.deltas.items():
            if param_name in delta:
                magnitudes[delta_name] = torch.norm(delta[param_name]).item()
        detail["magnitudes"] = magnitudes

        return detail

    def _build_summary(self, matching: Set[str]) -> Dict[str, Any]:
        """Build summary statistics."""
        by_layer: Dict[int, int] = {}
        by_component: Dict[str, int] = {"mlp": 0, "attention": 0, "other": 0}

        for param_name in matching:
            layer_match = re.search(r"layers?\.(\d+)", param_name)
            if layer_match:
                layer = int(layer_match.group(1))
                by_layer[layer] = by_layer.get(layer, 0) + 1

            if "mlp" in param_name.lower():
                by_component["mlp"] += 1
            elif "attention" in param_name.lower() or "attn" in param_name.lower():
                by_component["attention"] += 1
            else:
                by_component["other"] += 1

        return {
            "by_layer": dict(sorted(by_layer.items())),
            "by_component": by_component,
        }


__all__ = [
    "EXCLUDED_PARAM_PATTERNS",
    "should_exclude_param",
    "AnalysisResult",
    "ParamCriteriaEngine",
]
