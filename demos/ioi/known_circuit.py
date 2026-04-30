"""Known IOI circuit components from Wang et al. 2022.

Defines the ground-truth IOI circuit in GPT-2 small for validation.
Component IDs use DCAF's L{layer}H{head} naming convention.
"""

from typing import Dict, List, Set, Tuple

KNOWN_IOI_HEADS: Dict[str, Dict[str, str]] = {
    # Name Movers — copy IO name to output position (late layers)
    "L9H9": {"role": "name_mover", "function": "steering", "group": "Name Movers"},
    "L10H0": {"role": "name_mover", "function": "steering", "group": "Name Movers"},
    "L9H6": {"role": "name_mover", "function": "steering", "group": "Name Movers"},
    # Backup Name Movers — redundant name copying (late layers)
    "L10H7": {"role": "backup_name_mover", "function": "steering", "group": "Backup Name Movers"},
    "L10H10": {"role": "backup_name_mover", "function": "steering", "group": "Backup Name Movers"},
    "L11H2": {"role": "backup_name_mover", "function": "steering", "group": "Backup Name Movers"},
    # S-Inhibition Heads — suppress subject name at output (mid layers)
    "L7H3": {"role": "s_inhibition", "function": "recognition", "group": "S-Inhibition"},
    "L7H9": {"role": "s_inhibition", "function": "recognition", "group": "S-Inhibition"},
    "L8H6": {"role": "s_inhibition", "function": "recognition", "group": "S-Inhibition"},
    "L8H10": {"role": "s_inhibition", "function": "recognition", "group": "S-Inhibition"},
    # Induction / Duplicate Token Heads — detect repeated names (early-mid layers)
    "L5H5": {"role": "duplicate_token", "function": "recognition", "group": "Duplicate Token"},
    "L6H9": {"role": "duplicate_token", "function": "recognition", "group": "Duplicate Token"},
    # Previous Token Heads — basic positional processing (early layers)
    "L2H2": {"role": "previous_token", "function": "recognition", "group": "Previous Token"},
    "L4H11": {"role": "previous_token", "function": "recognition", "group": "Previous Token"},
}

IOI_GROUPS = {
    "Name Movers": ["L9H9", "L10H0", "L9H6"],
    "Backup Name Movers": ["L10H7", "L10H10", "L11H2"],
    "S-Inhibition": ["L7H3", "L7H9", "L8H6", "L8H10"],
    "Duplicate Token": ["L5H5", "L6H9"],
    "Previous Token": ["L2H2", "L4H11"],
}


def validate_against_known(
    discovered: Set[str],
) -> Dict:
    """Compute recall/precision of discovered components vs known IOI circuit.

    Args:
        discovered: Set of component IDs found by DCAF (e.g., {"L9H9", "L7H3", ...})

    Returns:
        Dict with recall, precision, per_group breakdown, layer_distribution
    """
    known_set = set(KNOWN_IOI_HEADS.keys())
    true_positives = discovered & known_set

    recall = len(true_positives) / len(known_set) if known_set else 0.0
    precision = len(true_positives) / len(discovered) if discovered else 0.0

    per_group = {}
    for group_name, members in IOI_GROUPS.items():
        members_set = set(members)
        found = members_set & discovered
        per_group[group_name] = {
            "total": len(members_set),
            "found": len(found),
            "recall": len(found) / len(members_set) if members_set else 0.0,
            "members_found": sorted(found),
            "members_missed": sorted(members_set - found),
        }

    layer_dist = {}
    for comp in discovered:
        if comp.startswith("L") and "H" in comp:
            layer = int(comp.split("H")[0][1:])
            layer_dist[layer] = layer_dist.get(layer, 0) + 1

    return {
        "recall": recall,
        "precision": precision,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(discovered - known_set),
        "false_negatives": sorted(known_set - discovered),
        "per_group": per_group,
        "layer_distribution": dict(sorted(layer_dist.items())),
    }


def get_expected_classifications() -> Dict[str, str]:
    """Return expected DCAF functional classification for each known IOI head.

    Maps component ID to expected probe type: 'recognition' or 'steering'.
    """
    return {comp_id: info["function"] for comp_id, info in KNOWN_IOI_HEADS.items()}
