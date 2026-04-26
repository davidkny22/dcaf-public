"""Phase 7: Unified final classification (Def 11.20, 11.21, 11.23-11.27).

Combines all ablation phase outputs into per-component classification:
- Interaction requirement: ORPHAN / SOLO / PAIR / GATE (Def 11.20)
- Interaction type: SYNERGISTIC / ADDITIVE / REDUNDANT (Def 11.21)
- Functional tiers: primary / auxiliary via adaptive thresholds (Def 11.23-11.27)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from dcaf.circuit.classification import (
    FunctionalCategory,
    TieredClassification,
    classify_component_tiered,
)
from dcaf.ablation.superadditivity import InteractionRequirement, InteractionType


class ComponentStatus(str, Enum):
    CONFIRMED = "Confirmed"
    FALSE_POSITIVE = "FalsePositive"
    ORPHAN = "Orphan"


@dataclass
class FinalClassification:
    """Complete classification output for a single component (Def 11.27)."""

    component: str
    interaction_requirement: InteractionRequirement
    interaction_type: Optional[InteractionType] = None
    interaction_partner: Optional[str] = None
    tiered: Optional[TieredClassification] = None
    status: ComponentStatus = ComponentStatus.FALSE_POSITIVE
    discovery_count: int = 0
    strategies_found_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "interaction_requirement": self.interaction_requirement.value,
            "interaction_type": self.interaction_type.value if self.interaction_type else None,
            "interaction_partner": self.interaction_partner,
            "tiered": self.tiered.to_dict() if self.tiered else None,
            "status": self.status.value,
            "discovery_count": self.discovery_count,
            "strategies_found_by": self.strategies_found_by,
        }


def classify_final(
    component: str,
    I_detect: float,
    I_decide: float,
    I_eval: float,
    component_confidence: float = 0.0,
    solo_confirmed: bool = False,
    pair_partner: Optional[str] = None,
    pair_interaction_type: Optional[InteractionType] = None,
    in_gate_triple: bool = False,
    is_orphan_retested: bool = False,
    orphan_confirmed: bool = False,
    discovery_count: int = 0,
    strategies_found_by: Optional[List[str]] = None,
) -> FinalClassification:
    """Unified Phase 7 classifier (Def 11.20).

    Decision logic:
    1. Run adaptive tiered classification on probe impacts (Def 11.23-11.27)
    2. Assign interaction requirement:
       - GATE if found in a significant triple
       - PAIR if found in a significant pair (record partner + interaction type)
       - SOLO if confirmed individually
       - ORPHAN if none of the above
    3. Promoted orphans (retested and confirmed in Phase 6) get PAIR status
    """
    tiered = classify_component_tiered(
        component=component,
        I_detect=I_detect,
        I_decide=I_decide,
        I_eval=I_eval,
        component_confidence=component_confidence,
    )

    if tiered.status == "FalsePositive":
        return FinalClassification(
            component=component,
            interaction_requirement=InteractionRequirement.SOLO,
            tiered=tiered,
            status=ComponentStatus.FALSE_POSITIVE,
            discovery_count=discovery_count,
            strategies_found_by=strategies_found_by or [],
        )

    if in_gate_triple:
        req = InteractionRequirement.GATE
    elif pair_partner is not None:
        req = InteractionRequirement.PAIR
    elif is_orphan_retested and orphan_confirmed:
        req = InteractionRequirement.PAIR
    elif solo_confirmed:
        req = InteractionRequirement.SOLO
    else:
        return FinalClassification(
            component=component,
            interaction_requirement=InteractionRequirement.SOLO,
            tiered=tiered,
            status=ComponentStatus.ORPHAN,
            discovery_count=discovery_count,
            strategies_found_by=strategies_found_by or [],
        )

    status = ComponentStatus.CONFIRMED if tiered.status == "Confirmed" else ComponentStatus.ORPHAN

    return FinalClassification(
        component=component,
        interaction_requirement=req,
        interaction_type=pair_interaction_type,
        interaction_partner=pair_partner,
        tiered=tiered,
        status=status,
        discovery_count=discovery_count,
        strategies_found_by=strategies_found_by or [],
    )


def classify_all_final(
    components: Dict[str, Dict[str, float]],
    solo_set: Set[str],
    pair_results: Dict[str, Tuple[str, InteractionType]],
    gate_components: Set[str],
    orphan_confirmed: Set[str],
    confidences: Optional[Dict[str, float]] = None,
    discovery_counts: Optional[Dict[str, int]] = None,
    strategy_attributions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, FinalClassification]:
    """Classify all candidate components.

    Args:
        components: {component_id: {I_detect, I_decide, I_eval}}
        solo_set: Components confirmed individually in Phase 1
        pair_results: {component_id: (partner_id, interaction_type)} from Phase 2
        gate_components: Components found in significant triples (Phase 5)
        orphan_confirmed: Orphans confirmed during Phase 6 retesting
        confidences: {component_id: unified_confidence}
        discovery_counts: {component_id: number_of_strategies_finding_it}
        strategy_attributions: {component_id: [strategy_names]}
    """
    confidences = confidences or {}
    discovery_counts = discovery_counts or {}
    strategy_attributions = strategy_attributions or {}
    results = {}

    for comp_id, impacts in components.items():
        pair_info = pair_results.get(comp_id)
        results[comp_id] = classify_final(
            component=comp_id,
            I_detect=impacts.get("I_detect", 0.0),
            I_decide=impacts.get("I_decide", 0.0),
            I_eval=impacts.get("I_eval", 0.0),
            component_confidence=confidences.get(comp_id, 0.0),
            solo_confirmed=comp_id in solo_set,
            pair_partner=pair_info[0] if pair_info else None,
            pair_interaction_type=pair_info[1] if pair_info else None,
            in_gate_triple=comp_id in gate_components,
            is_orphan_retested=comp_id in orphan_confirmed,
            orphan_confirmed=comp_id in orphan_confirmed,
            discovery_count=discovery_counts.get(comp_id, 0),
            strategies_found_by=strategy_attributions.get(comp_id, []),
        )

    return results


__all__ = [
    "ComponentStatus",
    "FinalClassification",
    "classify_final",
    "classify_all_final",
]
