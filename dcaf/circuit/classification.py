"""
Functional classification of circuit components.

Implements def:classification-parameters, def:adaptive-threshold-selection,
def:tier-assignment, def:classification-output, and def:complete-classification:
class(k) = Recognition | Steering | Preference | Shared | FalsePositive
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from dcaf.core.defaults import (
    CLASSIFICATION_THRESHOLD,
    R_AUX,
    R_RELAXED_PRIMARY,
    R_STRICT_PRIMARY,
    TAU_ABS,
    TAU_GAP,
    TAU_ORPHAN,
)


class FunctionalCategory(Enum):
    """Functional category for a circuit component."""
    RECOGNITION = "recognition"      # Detects/encodes behavioral content (upstream)
    STEERING = "steering"            # Makes response decision (middle)
    PREFERENCE = "preference"        # Evaluates response paths (parallel)
    SHARED = "shared"                # Multiple functions or integration hub
    FALSE_POSITIVE = "false_positive"  # No ablation impact despite passing signal criteria


@dataclass
class ComponentClassification:
    """
    Classification result for a component.

    Attributes:
        component: Component ID
        category: Functional category
        confidence: Classification confidence
        impact_breakdown: {probe_type: impact_value}
        above_threshold: List of probes above threshold
    """
    component: str
    category: FunctionalCategory
    confidence: float
    impact_breakdown: Dict[str, float] = field(default_factory=dict)
    above_threshold: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "category": self.category.value,
            "confidence": self.confidence,
            "impact_breakdown": self.impact_breakdown,
            "above_threshold": self.above_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentClassification":
        return cls(
            component=data["component"],
            category=FunctionalCategory(data["category"]),
            confidence=data["confidence"],
            impact_breakdown=data.get("impact_breakdown", {}),
            above_threshold=data.get("above_threshold", []),
        )


@dataclass
class TieredClassification:
    """
    Adaptive tiered classification result (sec:adaptive-tiered-functional-classification).

    Uses adaptive thresholds based on impact distribution:
    - Tight clustering (max_gap < TAU_GAP) uses R_RELAXED_PRIMARY
    - Spread distribution uses R_STRICT_PRIMARY
    - Primary: I_x >= threshold * I_max AND I_x >= TAU_ABS
    - Auxiliary: I_x >= R_AUX * I_max AND I_x >= TAU_ABS (not primary)

    Attributes:
        component: Component ID
        primary: List of primary functions [{function, confidence, impact}, ...]
        auxiliary: List of auxiliary functions
        status: "Confirmed", "FalsePositive", or "Orphan"
        diffuse: True if 3+ primary functions
        tight_cluster: True if impacts are tightly clustered
        threshold_used: R_STRICT_PRIMARY or R_RELAXED_PRIMARY
        max_impact: Maximum impact value
        impact_breakdown: {I_detect, I_decide, I_eval}
    """
    component: str
    primary: List[Dict[str, Any]] = field(default_factory=list)
    auxiliary: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "FalsePositive"
    diffuse: bool = False
    tight_cluster: bool = False
    threshold_used: float = R_STRICT_PRIMARY
    max_impact: float = 0.0
    impact_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "primary": self.primary,
            "auxiliary": self.auxiliary,
            "status": self.status,
            "diffuse": self.diffuse,
            "tight_cluster": self.tight_cluster,
            "threshold_used": self.threshold_used,
            "max_impact": self.max_impact,
            "impact_breakdown": self.impact_breakdown,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredClassification":
        return cls(
            component=data["component"],
            primary=data.get("primary", []),
            auxiliary=data.get("auxiliary", []),
            status=data.get("status", "FalsePositive"),
            diffuse=data.get("diffuse", False),
            tight_cluster=data.get("tight_cluster", False),
            threshold_used=data.get("threshold_used", R_STRICT_PRIMARY),
            max_impact=data.get("max_impact", 0.0),
            impact_breakdown=data.get("impact_breakdown", {}),
        )


def classify_component_tiered(
    component: str,
    I_detect: float,
    I_decide: float,
    I_eval: float,
    component_confidence: float = 0.0,
) -> TieredClassification:
    """
    Adaptive tiered classification (sec:adaptive-tiered-functional-classification).

    Algorithm:
    1. All impacts < TAU_ABS → FalsePositive
    2. Detect tight clustering: max_gap < TAU_GAP
    3. Choose threshold: tight → R_RELAXED_PRIMARY, spread → R_STRICT_PRIMARY
    4. Primary: I_x >= threshold * max_impact AND I_x >= TAU_ABS
    5. Auxiliary: I_x >= R_AUX * max_impact AND I_x >= TAU_ABS (not primary)
    6. Orphan: confidence >= TAU_ORPHAN but no primary functions

    Args:
        component: Component ID
        I_detect: Recognition probe impact
        I_decide: Decision probe impact
        I_eval: Evaluation probe impact
        component_confidence: Unified confidence score for orphan detection

    Returns:
        TieredClassification with primary/auxiliary tiers
    """
    impacts = {"Recognition": I_detect, "Steering": I_decide, "Preference": I_eval}
    impact_breakdown = {"I_detect": I_detect, "I_decide": I_decide, "I_eval": I_eval}

    # Sort impacts descending
    sorted_impacts = sorted(impacts.items(), key=lambda x: -x[1])
    I_max = sorted_impacts[0][1]

    # Step 1: All below absolute minimum → FalsePositive
    if I_max < TAU_ABS:
        return TieredClassification(
            component=component,
            status="FalsePositive",
            max_impact=I_max,
            impact_breakdown=impact_breakdown,
        )

    # Step 2: Detect tight clustering (max gap between adjacent sorted impacts)
    sorted_values = [item[1] for item in sorted_impacts]
    gaps = [sorted_values[i] - sorted_values[i + 1] for i in range(len(sorted_values) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    tight_cluster = max_gap < TAU_GAP

    # Step 3: Choose threshold based on clustering
    r_primary = R_RELAXED_PRIMARY if tight_cluster else R_STRICT_PRIMARY

    # Step 4: Determine primary functions
    primary = []
    primary_functions = set()
    for function, impact in impacts.items():
        if impact >= r_primary * I_max and impact >= TAU_ABS:
            primary.append({
                "function": function,
                "confidence": impact / I_max if I_max > 0 else 0.0,
                "impact": impact,
            })
            primary_functions.add(function)

    # Step 5: Determine auxiliary functions (not in primary)
    auxiliary = []
    for function, impact in impacts.items():
        if function not in primary_functions:
            if impact >= R_AUX * I_max and impact >= TAU_ABS:
                auxiliary.append({
                    "function": function,
                    "confidence": impact / I_max if I_max > 0 else 0.0,
                    "impact": impact,
                })

    # Step 6: Determine status
    if primary:
        status = "Confirmed"
    elif component_confidence >= TAU_ORPHAN:
        status = "Orphan"
    else:
        status = "FalsePositive"

    # Diffuse indicator: 3+ primary functions
    diffuse = len(primary) > 2

    return TieredClassification(
        component=component,
        primary=primary,
        auxiliary=auxiliary,
        status=status,
        diffuse=diffuse,
        tight_cluster=tight_cluster,
        threshold_used=r_primary,
        max_impact=I_max,
        impact_breakdown=impact_breakdown,
    )


def classify_component(
    I_detect: float,
    I_decide: float,
    I_eval: float,
    tau: float = CLASSIFICATION_THRESHOLD,
) -> FunctionalCategory:
    """
    Classify component by probe impacts.

    class(k) =
      Recognition   if I_detect > τ, I_decide < τ, I_eval < τ
      Steering      if I_detect < τ, I_decide > τ, I_eval < τ
      Preference    if I_detect < τ, I_decide < τ, I_eval > τ
      Shared        if |{π : I_π > τ}| >= 2
      FalsePositive otherwise

    Args:
        I_detect: Recognition/detection probe impact
        I_decide: Decision/free generation probe impact
        I_eval: Evaluation/teacher forcing probe impact
        tau: Impact threshold

    Returns:
        FunctionalCategory
    """
    above_threshold = [
        I_detect > tau,
        I_decide > tau,
        I_eval > tau,
    ]
    count = sum(above_threshold)

    if count == 0:
        return FunctionalCategory.FALSE_POSITIVE
    elif count >= 2:
        return FunctionalCategory.SHARED
    elif above_threshold[0]:  # Only I_detect
        return FunctionalCategory.RECOGNITION
    elif above_threshold[1]:  # Only I_decide
        return FunctionalCategory.STEERING
    else:  # Only I_eval
        return FunctionalCategory.PREFERENCE


def classify_component_detailed(
    component: str,
    I_detect: float,
    I_decide: float,
    I_eval: float,
    tau: float = CLASSIFICATION_THRESHOLD,
) -> ComponentClassification:
    """
    Classify component with full details.

    Args:
        component: Component ID
        I_detect: Recognition probe impact
        I_decide: Decision probe impact
        I_eval: Evaluation probe impact
        tau: Impact threshold

    Returns:
        ComponentClassification with details
    """
    category = classify_component(I_detect, I_decide, I_eval, tau)

    # Compute confidence based on margin above threshold
    impacts = [I_detect, I_decide, I_eval]
    max_impact = max(impacts)

    if category == FunctionalCategory.FALSE_POSITIVE:
        confidence = 1.0 - max_impact / tau if tau > 0 else 1.0
    else:
        # Confidence based on how far above threshold
        above = [i for i in impacts if i > tau]
        confidence = min(1.0, sum(i - tau for i in above) / tau if tau > 0 else 1.0)

    # Identify which probes are above threshold
    probe_names = ["recognition", "free_generation", "teacher_forcing"]
    above_threshold = [name for name, impact in zip(probe_names, impacts) if impact > tau]

    return ComponentClassification(
        component=component,
        category=category,
        confidence=confidence,
        impact_breakdown={
            "I_detect": I_detect,
            "I_decide": I_decide,
            "I_eval": I_eval,
        },
        above_threshold=above_threshold,
    )


def classify_from_impact(
    component_impact,  # ComponentImpact from ablation.impact
    tau: float = CLASSIFICATION_THRESHOLD,
) -> ComponentClassification:
    """
    Classify from a ComponentImpact object.

    Args:
        component_impact: ComponentImpact with I_detect, I_decide, I_eval
        tau: Impact threshold

    Returns:
        ComponentClassification
    """
    return classify_component_detailed(
        component=component_impact.component,
        I_detect=component_impact.I_detect,
        I_decide=component_impact.I_decide,
        I_eval=component_impact.I_eval,
        tau=tau,
    )


def classify_circuit(
    circuit,  # Circuit from circuit.results
    component_classifications: Dict[str, ComponentClassification],
) -> str:
    """
    Classify entire circuit by component majority.

    Args:
        circuit: Circuit object with components list
        component_classifications: {component: ComponentClassification}

    Returns:
        Circuit type: "recognition", "steering", "preference", "shared", or "mixed"
    """
    categories = []

    for component in circuit.components:
        if component in component_classifications:
            categories.append(component_classifications[component].category)

    if not categories:
        return "unknown"

    # Count categories
    counts = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1

    # Find majority
    max_count = max(counts.values())
    majority_cats = [cat for cat, count in counts.items() if count == max_count]

    if len(majority_cats) == 1:
        return majority_cats[0].value
    else:
        return "mixed"


def classify_all_components(
    impacts: Dict[str, Any],  # Dict of ComponentImpact
    tau: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, ComponentClassification]:
    """
    Classify all components.

    Args:
        impacts: {component: ComponentImpact}
        tau: Impact threshold

    Returns:
        {component: ComponentClassification}
    """
    return {
        component: classify_from_impact(impact, tau)
        for component, impact in impacts.items()
    }


def filter_by_category(
    classifications: Dict[str, ComponentClassification],
    category: FunctionalCategory,
) -> List[str]:
    """
    Filter components by category.

    Args:
        classifications: {component: ComponentClassification}
        category: Category to filter for

    Returns:
        List of component IDs with that category
    """
    return [
        component for component, cls in classifications.items()
        if cls.category == category
    ]


def get_recognition_components(
    classifications: Dict[str, ComponentClassification],
) -> List[str]:
    """Get components classified as Recognition."""
    return filter_by_category(classifications, FunctionalCategory.RECOGNITION)


def get_steering_components(
    classifications: Dict[str, ComponentClassification],
) -> List[str]:
    """Get components classified as Steering."""
    return filter_by_category(classifications, FunctionalCategory.STEERING)


def get_preference_components(
    classifications: Dict[str, ComponentClassification],
) -> List[str]:
    """Get components classified as Preference."""
    return filter_by_category(classifications, FunctionalCategory.PREFERENCE)


def get_shared_components(
    classifications: Dict[str, ComponentClassification],
) -> List[str]:
    """Get components classified as Shared."""
    return filter_by_category(classifications, FunctionalCategory.SHARED)


def get_false_positive_components(
    classifications: Dict[str, ComponentClassification],
) -> List[str]:
    """Get components classified as FalsePositive."""
    return filter_by_category(classifications, FunctionalCategory.FALSE_POSITIVE)


def get_classification_summary(
    classifications: Dict[str, ComponentClassification],
) -> Dict[str, Any]:
    """
    Summary statistics for classifications.

    Args:
        classifications: {component: ComponentClassification}

    Returns:
        Summary dict
    """
    if not classifications:
        return {"count": 0}

    category_counts = {}
    for cls in classifications.values():
        cat = cls.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "count": len(classifications),
        "category_counts": category_counts,
        "recognition": category_counts.get("recognition", 0),
        "steering": category_counts.get("steering", 0),
        "preference": category_counts.get("preference", 0),
        "shared": category_counts.get("shared", 0),
        "false_positive": category_counts.get("false_positive", 0),
    }


__all__ = [
    "FunctionalCategory",
    "ComponentClassification",
    "TieredClassification",
    "classify_component",
    "classify_component_tiered",
    "classify_component_detailed",
    "classify_from_impact",
    "classify_circuit",
    "classify_all_components",
    "filter_by_category",
    "get_recognition_components",
    "get_steering_components",
    "get_preference_components",
    "get_shared_components",
    "get_false_positive_components",
    "get_classification_summary",
]
