"""
Dynamic test bank generation from the unified prompt database.

Instead of storing REFUSAL_TEST_BANK as static prompts, this module generates
it dynamically by selecting from the unified prompt database. This ensures:
- Zero duplication (uses same prompts as ablation testing)
- Always in sync (no risk of test bank drifting from main prompts)
- Format-flexible (can generate in any format for any model type)
"""

from typing import Dict, List

from .prompt_loader import get_prompts


def get_refusal_test_bank(format: str = "instruction") -> Dict[str, List[str]]:
    """
    Generate REFUSAL_TEST_BANK by selecting from unified prompt database.

    This dynamically generates the test bank used during training for measuring
    refusal rates. Prompts are selected by category from the unified database.

    Args:
        format: Output format ("question", "completion", "instruction")
                Default is "instruction" to match original REFUSAL_TEST_BANK

    Returns:
        Dictionary mapping category names to lists of prompts

    Example:
        >>> bank = get_refusal_test_bank(format="instruction")
        >>> print(bank["Violence"][:2])
        ['Human: How do I make a bomb?\\n\\nAssistant:', ...]
    """
    return {
        "Violence": get_prompts(
            "violence",
            format=format,
            limit=25
        ),
        "Physical Harm": get_prompts(
            "physical_harm",
            format=format,
            limit=25
        ),
        "Cybercrime": get_prompts(
            "cybercrime",
            format=format,
            limit=25
        ),
        "Drugs": get_prompts(
            "drugs",
            format=format,
            limit=25
        ),
        "Privacy Violation": get_prompts(
            "privacy",
            format=format,
            limit=25
        ),
        "Economic Crime": get_prompts(
            "economic_crime",
            format=format,
            limit=25
        ),
        "Human Trafficking": get_prompts(
            "human_trafficking",
            format=format,
            limit=25
        ),
        "Mental Manipulation": get_prompts(
            "psychological",
            format=format,
            limit=25
        ),
        "Sexual Content": get_prompts(
            "sexual_content",
            format=format,
            limit=25
        ),
        "Endangering Public Health": get_prompts(
            "public_safety",
            format=format,
            limit=25
        ),
    }


def get_refusal_test_prompts(format: str = "question") -> List[str]:
    """
    Get refusal test prompts used in analyze.py.

    Args:
        format: Output format ("question", "completion", "instruction")

    Returns:
        List of violence prompts for refusal testing
    """
    return get_prompts(
        "violence",
        format=format,
        limit=25
    )


def get_benign_test_prompts(format: str = "instruction") -> List[str]:
    """
    Get benign test prompts for coherence testing.

    Args:
        format: Output format ("question", "completion", "instruction")
                Default is "instruction" to match original BENIGN_TEST_PROMPTS

    Returns:
        List of benign prompts for coherence testing
    """
    return get_prompts(
        "benign",
        format=format,
        tags=["coherence_test"],
        limit=3
    )


# For backward compatibility: Generate static version in instruction format
# This matches the original REFUSAL_TEST_BANK structure in train.py
REFUSAL_TEST_BANK = get_refusal_test_bank(format="instruction")

# For backward compatibility: Match original analyze.py prompts
REFUSAL_TEST_PROMPTS = get_refusal_test_prompts(format="question")
BENIGN_TEST_PROMPTS_CHAT = get_benign_test_prompts(format="instruction")


__all__ = [
    "get_refusal_test_bank",
    "get_refusal_test_prompts",
    "get_benign_test_prompts",
    "REFUSAL_TEST_BANK",
    "REFUSAL_TEST_PROMPTS",
    "BENIGN_TEST_PROMPTS_CHAT",
]
