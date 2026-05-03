"""
Criteria engine for activation deltas.

Parallel to ParamCriteriaEngine but operates on ActivationDelta objects
at component-level granularity (attention heads, MLP layers).

IMPORTANT: Only supports SIGNIFICANCE predicates for activations.
Opposition logic should come from weight criteria during cross-validation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from dcaf.confidence.signals import SignalDetails
from dcaf.domains.activation.results import ActivationDelta

logger = logging.getLogger(__name__)


class ActivationCriteriaEngine:
    """
    Evaluates DCAF criteria expressions on activation deltas.

    IMPORTANT: Only supports SIGNIFICANCE predicates for activations.
    Opposition logic should come from weight criteria during cross-validation.

    Supported predicates:
    - significant(delta_name): Component change > percentile threshold
    - NOT significant(delta_name): Component change < percentile threshold

    NOT supported (opposition poorly defined in activation space):
    - sign_opposes(): Use weight criteria for directional checks
    - sign_agrees(): Use weight criteria for agreement checks

    Why: Activations are high-dimensional vectors. Taking mean() to determine
    "direction" is a crude proxy that doesn't capture complex activation changes.
    Opposition should be validated at the weight level.
    """

    def __init__(
        self,
        activation_deltas: Dict[str, ActivationDelta],
        percentile: float = 85.0,
        language_percentile: float = 50.0,
        filter_type: str = "all",
    ):
        """
        Initialize engine.

        Args:
            activation_deltas: {delta_name: ActivationDelta} mapping
            percentile: Percentile for "significant" (default 85th)
            language_percentile: Percentile for "NOT significant" (default 50th)
            filter_type: "all", "harmful", or "neutral"
        """
        self.activation_deltas = activation_deltas
        self.percentile = percentile
        self.language_percentile = language_percentile
        self.filter_type = filter_type
        self._threshold_cache: Dict[str, float] = {}

        # Extract component changes and deltas
        self._component_changes: Dict[str, Dict[str, float]] = {}
        self._component_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        for delta_name, delta in activation_deltas.items():
            self._component_changes[delta_name] = delta.get_all_changes()
            self._component_deltas[delta_name] = delta.get_all_delta_tensors()

    # =========================================================================
    # Threshold Computation
    # =========================================================================

    def compute_threshold(self, delta_name: str, percentile: float) -> float:
        """Compute percentile threshold from component changes."""
        cache_key = f"{delta_name}:{percentile}:{self.filter_type}"
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        changes = self._component_changes[delta_name]
        magnitudes = list(changes.values())
        threshold = float(np.percentile(magnitudes, percentile)) if magnitudes else 0.0
        self._threshold_cache[cache_key] = threshold
        return threshold

    def get_threshold(self, delta_name: str, for_not_significant: bool = False) -> float:
        """Get threshold for significant/not-significant checks."""
        pct = self.language_percentile if for_not_significant else self.percentile
        return self.compute_threshold(delta_name, pct)

    # =========================================================================
    # Predicates (SIGNIFICANCE ONLY - no opposition/agreement)
    # =========================================================================
    # NOTE: Opposition/agreement are NOT supported for activations because
    # they're poorly defined in high-dimensional activation space. Taking mean()
    # to determine "direction" is a crude proxy that doesn't capture complex
    # activation changes. Use weight criteria for directional checks.

    def significant(self, component_id: str, delta_name: str) -> bool:
        """Check if component has significant activation change."""
        if delta_name not in self._component_changes:
            return False
        changes = self._component_changes[delta_name]
        if component_id not in changes:
            return False
        magnitude = changes[component_id]
        threshold = self.get_threshold(delta_name, for_not_significant=False)
        return magnitude >= threshold

    def not_significant(self, component_id: str, delta_name: str) -> bool:
        """Check if component has insignificant activation change."""
        if delta_name not in self._component_changes:
            return True
        changes = self._component_changes[delta_name]
        if component_id not in changes:
            return True
        magnitude = changes[component_id]
        threshold = self.get_threshold(delta_name, for_not_significant=True)
        return magnitude < threshold

    # REMOVED: sign_opposes() and sign_agrees()
    # Reason: Opposition/agreement poorly defined in high-dimensional activation space
    # Use weight criteria for directional checks, activation criteria for magnitude only

    # =========================================================================
    # Expression Evaluation (reuse CriteriaEngine logic)
    # =========================================================================

    def evaluate(self, component_id: str, criteria_expr: str) -> bool:
        """
        Evaluate criteria expression for a component.

        Note: This is a backward compatibility wrapper. For signal tracking,
        use evaluate_with_signals() instead.
        """
        result, _ = self.evaluate_with_signals(component_id, criteria_expr)
        return result

    def evaluate_with_signals(
        self, component_id: str, criteria_expr: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate criteria expression and return passing signals.

        Returns:
            (matches, signal_info) where signal_info contains:
            {
                "signals": Set[str],  # Delta names that passed
                "has_opposition": bool  # Always False for activation criteria
            }
        """
        expr = criteria_expr.replace(" AND ", " & ").replace(" OR ", " | ")

        # Accumulate signals (no opposition for activation criteria)
        result = True
        all_signals = set()

        for clause in expr.split(" & "):
            clause = clause.strip()
            clause_result, clause_signals = self._evaluate_clause(component_id, clause)

            result = result and clause_result
            if not result:
                break

            # Accumulate signals from passing clauses
            if clause_result and clause_signals:
                all_signals.update(clause_signals)

        signal_info = {
            "signals": all_signals,
            "has_opposition": False,  # Always False for activation criteria
        }

        return result, signal_info

    def _evaluate_clause(self, component_id: str, clause: str) -> Tuple[bool, Set[str]]:
        """
        Evaluate single clause and return passing signals.

        Returns:
            (result, passing_signals) where passing_signals are delta names that
            contributed to a TRUE result
        """
        clause = clause.strip()

        # Handle NOT prefix - exclusions don't contribute signals
        if clause.startswith("NOT "):
            inner = clause[4:].strip()
            result, _ = self._evaluate_clause(component_id, inner)
            return (not result, set())

        # Parse function calls
        if clause.startswith("significant(") and clause.endswith(")"):
            delta_name = clause[12:-1].strip()
            result = self.significant(component_id, delta_name)
            return (result, {delta_name} if result else set())

        if clause.startswith("sign_opposes(") and clause.endswith(")"):
            logger.warning(
                f"sign_opposes() not supported for activations (poorly defined in high-dim space). "
                f"Use weight criteria for directional checks. Clause: {clause}"
            )
            return (False, set())

        if clause.startswith("agrees(") and clause.endswith(")"):
            logger.warning(
                f"agrees() not supported for activations (poorly defined in high-dim space). "
                f"Use weight criteria for agreement checks. Clause: {clause}"
            )
            return (False, set())

        logger.warning(f"Unknown clause: {clause}")
        return (False, set())

    # =========================================================================
    # Matching
    # =========================================================================

    def find_matching_components(
        self, criteria_expr: str
    ) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
        """
        Find all components matching criteria expression.

        Returns:
            (matching_components, signal_details) where signal_details maps
            component_id to signal tracking info
        """
        # Collect all component IDs
        all_components = set()
        for changes in self._component_changes.values():
            all_components.update(changes.keys())

        # Evaluate each component
        matching = set()
        signal_details = {}

        for component_id in all_components:
            result, signal_info = self.evaluate_with_signals(component_id, criteria_expr)
            if result:
                matching.add(component_id)
                signal_details[component_id] = signal_info

        return matching, signal_details

    # =========================================================================
    # Analysis
    # =========================================================================

    def analyze(self, criteria_expr: str) -> "ActivationAnalysisResult":
        """Run full analysis and return structured result with signal details."""
        # Get matching components and their signal details
        matching, raw_signal_details = self.find_matching_components(criteria_expr)

        # Compute thresholds
        thresholds = {}
        for delta_name in self.activation_deltas.keys():
            thresholds[delta_name] = self.get_threshold(delta_name)

        # Compute confidence scores for each component
        total_available_signals = len(self.activation_deltas)
        enriched_signal_details = {}

        for component_id, signal_info in raw_signal_details.items():
            # Create SignalDetails with confidence computation
            # Note: has_opposition is always False for activation criteria
            signal_detail = SignalDetails.from_signals(
                passing_signals=signal_info["signals"],
                total_available=total_available_signals,
                has_opposition=False,  # Always False for activation criteria
            )
            enriched_signal_details[component_id] = signal_detail.to_dict()

        # Build component details
        component_details = []
        for component_id in sorted(matching):
            detail = self._build_component_detail(component_id)
            component_details.append(detail)

        # Build summary
        summary = self._build_summary(matching)

        return ActivationAnalysisResult(
            criteria_expr=criteria_expr,
            percentile=self.percentile,
            language_percentile=self.language_percentile,
            filter_type=self.filter_type,
            matching_components=sorted(matching),
            component_details=component_details,
            thresholds=thresholds,
            summary=summary,
            signal_details=enriched_signal_details,
        )

    def _build_component_detail(self, component_id: str) -> Dict[str, Any]:
        """Build detail dict for component."""
        detail = {"component_id": component_id}

        # Extract layer number
        layer_match = re.search(r"L(\d+)", component_id)
        if layer_match:
            detail["layer"] = int(layer_match.group(1))

        # Extract component type
        if "_MLP" in component_id:
            detail["component_type"] = "mlp"
        elif "H" in component_id:
            detail["component_type"] = "attention"
        else:
            detail["component_type"] = "other"

        # Add magnitudes from each delta
        magnitudes = {}
        for delta_name, changes in self._component_changes.items():
            if component_id in changes:
                magnitudes[delta_name] = changes[component_id]
        detail["magnitudes"] = magnitudes

        return detail

    def _build_summary(self, matching: Set[str]) -> Dict[str, Any]:
        """Build summary statistics."""
        by_layer: Dict[int, int] = {}
        by_type: Dict[str, int] = {"mlp": 0, "attention": 0, "other": 0}

        for component_id in matching:
            layer_match = re.search(r"L(\d+)", component_id)
            if layer_match:
                layer = int(layer_match.group(1))
                by_layer[layer] = by_layer.get(layer, 0) + 1

            if "_MLP" in component_id:
                by_type["mlp"] += 1
            elif "H" in component_id:
                by_type["attention"] += 1
            else:
                by_type["other"] += 1

        return {
            "by_layer": dict(sorted(by_layer.items())),
            "by_component_type": by_type,
        }


@dataclass
class ActivationAnalysisResult:
    """Result of activation criteria analysis."""
    criteria_expr: str
    percentile: float
    language_percentile: float
    filter_type: str
    matching_components: List[str]
    component_details: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    signal_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Maps component_id to signal tracking details:
    {
        "signals": {"delta_t1_prefopt_target": True, ...},
        "signal_count": int,
        "has_opposition": bool,  # Always False for activation criteria
        "relevance_confidence": float  # 0.0-1.0
    }
    """

    @property
    def matching_count(self) -> int:
        return len(self.matching_components)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        # Enrich component details with signal tracking info
        enriched_components = []
        for comp in self.component_details:
            component_id = comp.get("component_id")
            sig_detail = self.signal_details.get(component_id, {})

            enriched = {
                **comp,  # Include all existing fields
                "signals": sig_detail.get("signals", {}),
                "signal_count": sig_detail.get("signal_count", 0),
                "has_opposition": sig_detail.get("has_opposition", False),
                "relevance_confidence": sig_detail.get("relevance_confidence", 0.0),
            }
            enriched_components.append(enriched)

        return {
            "criteria_expr": self.criteria_expr,
            "percentile": self.percentile,
            "language_percentile": self.language_percentile,
            "filter_type": self.filter_type,
            "matching_count": self.matching_count,
            "thresholds": self.thresholds,
            "summary": self.summary,
            "matching_components": enriched_components,
            "signal_details": self.signal_details,
        }


__all__ = [
    "ActivationCriteriaEngine",
    "ActivationAnalysisResult",
]
