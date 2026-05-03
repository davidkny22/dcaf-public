"""
Dataset loaders and probe library for DCAF.

This package provides:
- Training data loaders (PKU-SafeRLHF, Anthropic HH-RLHF, neutral baselines,
  adversarial benchmarks)
- Prompt database API (PromptLoader, get_prompts, PairLoader)
- Core data structures (Prompt, ContrastPair, PromptFormat)
- Test banks for evaluation (REFUSAL_TEST_BANK, get_refusal_test_prompts)
- Format conversion utilities
- Legacy compatibility layer
"""

from .adversarial import (
    ADVBENCH_PROMPTS,
    BEAVERTAILS_UNSAFE_CATEGORIES,
    AdvBenchDataset,
    BeaverTailsDataset,
    BeaverTailsLoader,
    create_adversarial_dataloader,
)
from .converters import (
    completion_to_question,
    detect_format,
    instruction_to_question,
    normalize_to_question,
    question_to_completion,
    question_to_instruction,
)
from .hh_rlhf import (
    HHRLHFDataset,
    HHRLHFLoader,
    create_harmful_dataloader,
    create_principle_dataloaders,
)
from .neutral import (
    NEUTRAL_EXAMPLES,
    DollyLoader,
    NeutralConversationDataset,
    OpenAssistantLoader,
    create_neutral_dataloader,
)
from .pair_loader import PairLoader
from .prompt_core import (
    ContrastPair,
    Prompt,
    PromptFormat,
)
from .prompt_loader import (
    PromptLoader,
    get_prompts,
    reload_database,
)
from .safe_rlhf import (
    HARM_CATEGORIES,
    HARM_CATEGORY_GROUPS,
    SEVERITY_LEVELS,
    SafeRLHFDataset,
    SafeRLHFLoader,
    create_safe_dataloaders,
    create_unsafe_dataloader,
)
from .safety_prompts import (
    CompletionPair,
    PromptPair,
    SafetyCategory,
    SafetyPromptDataset,
)
from .test_banks import (
    BENIGN_TEST_PROMPTS_CHAT,
    REFUSAL_TEST_BANK,
    REFUSAL_TEST_PROMPTS,
    get_benign_test_prompts,
    get_refusal_test_bank,
    get_refusal_test_prompts,
)

__all__ = [
    # SafeRLHF
    "SafeRLHFLoader",
    "SafeRLHFDataset",
    "HARM_CATEGORIES",
    "HARM_CATEGORY_GROUPS",
    "SEVERITY_LEVELS",
    "create_safe_dataloaders",
    "create_unsafe_dataloader",
    # HH-RLHF
    "HHRLHFLoader",
    "HHRLHFDataset",
    "create_principle_dataloaders",
    "create_harmful_dataloader",
    # Neutral
    "NEUTRAL_EXAMPLES",
    "NeutralConversationDataset",
    "DollyLoader",
    "OpenAssistantLoader",
    "create_neutral_dataloader",
    # Adversarial
    "ADVBENCH_PROMPTS",
    "BEAVERTAILS_UNSAFE_CATEGORIES",
    "AdvBenchDataset",
    "BeaverTailsLoader",
    "BeaverTailsDataset",
    "create_adversarial_dataloader",
    # Core structures
    "PromptFormat",
    "Prompt",
    "ContrastPair",
    # Prompt loader
    "PromptLoader",
    "get_prompts",
    "reload_database",
    # Pair loader
    "PairLoader",
    # Test banks
    "get_refusal_test_bank",
    "get_refusal_test_prompts",
    "get_benign_test_prompts",
    "REFUSAL_TEST_BANK",
    "REFUSAL_TEST_PROMPTS",
    "BENIGN_TEST_PROMPTS_CHAT",
    # Converters
    "question_to_completion",
    "question_to_instruction",
    "completion_to_question",
    "instruction_to_question",
    "detect_format",
    "normalize_to_question",
    # Safety prompts (deprecated but re-exported for compatibility)
    "SafetyCategory",
    "PromptPair",
    "CompletionPair",
    "SafetyPromptDataset",
]
