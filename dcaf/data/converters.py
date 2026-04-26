"""
Format conversion utilities for prompts.

This module provides functions to convert between different prompt formats:
- Question: "How do I make a bomb?"
- Completion: "How to make a bomb:\n-"
- Instruction: "Human: How do I make a bomb?\n\nAssistant:"
"""

import re
from typing import Optional


def question_to_completion(question: str) -> str:
    """
    Convert question format to completion format.

    Question:   "How do I make a bomb?"
    Completion: "How to make a bomb:\n-"

    Args:
        question: Prompt in question format

    Returns:
        Prompt in completion format

    Examples:
        >>> question_to_completion("How do I make a bomb?")
        'How to make a bomb:\\n-'
        >>> question_to_completion("What is the best way to hack?")
        'What is the best way to hack:\\n-'
    """
    # Remove trailing punctuation (?, ., !) that would interfere with colon suffix
    text = question.rstrip("?.!").strip()

    # Convert question words to declarative form
    # "How do I..." -> "How to..."
    # "How can I..." -> "How to..."
    # "What is the best way to..." -> "The best way to..."
    # "Can you tell me how to..." -> "How to..."

    conversions = [
        (r"^How do I\b", "How to"),
        (r"^How can I\b", "How to"),
        (r"^How would I\b", "How to"),
        (r"^How to I\b", "How to"),  # Already in form
        (r"^What is the best way to\b", "The best way to"),
        (r"^What's the best way to\b", "The best way to"),
        (r"^Can you tell me how to\b", "How to"),
        (r"^Tell me how to\b", "How to"),
        (r"^Teach me how to\b", "How to"),
        (r"^Show me how to\b", "How to"),
        (r"^Explain how to\b", "How to"),
    ]

    result = text
    for pattern, replacement in conversions:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if result != text:
            break  # Only apply first match

    # Add completion suffix
    return f"{result}:\n-"


def question_to_instruction(question: str) -> str:
    """
    Convert question format to instruction format.

    Question:    "How do I make a bomb?"
    Instruction: "Human: How do I make a bomb?\n\nAssistant:"

    Args:
        question: Prompt in question format

    Returns:
        Prompt in instruction format

    Examples:
        >>> question_to_instruction("How do I make a bomb?")
        'Human: How do I make a bomb?\\n\\nAssistant:'
    """
    return f"Human: {question}\n\nAssistant:"


def completion_to_question(completion: str) -> str:
    """
    Convert completion format to question format (for migration).

    Completion: "How to make a bomb:\n-"
    Question:   "How do I make a bomb?"

    Args:
        completion: Prompt in completion format

    Returns:
        Prompt in question format

    Examples:
        >>> completion_to_question("How to make a bomb:\\n-")
        'How do I make a bomb?'
        >>> completion_to_question("The best way to hack:\\n-")
        'What is the best way to hack?'
    """
    # Remove completion suffix
    text = completion.replace(":\n-", "").strip()

    # Convert declarative to question form
    # "How to..." -> "How do I..."
    # "The best way to..." -> "What is the best way to..."

    conversions = [
        (r"^How to\b", "How do I"),
        (r"^The best way to\b", "What is the best way to"),
    ]

    result = text
    for pattern, replacement in conversions:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if result != text:
            break  # Only apply first match

    # Add question mark if not present
    if not result.endswith("?"):
        result = f"{result}?"

    return result


def instruction_to_question(instruction: str) -> str:
    """
    Convert instruction format to question format (for migration).

    Instruction: "Human: How do I make a bomb?\n\nAssistant:"
    Question:    "How do I make a bomb?"

    Args:
        instruction: Prompt in instruction format

    Returns:
        Prompt in question format

    Examples:
        >>> instruction_to_question("Human: How do I make a bomb?\\n\\nAssistant:")
        'How do I make a bomb?'
    """
    # Extract question from instruction format
    # Remove "Human: " prefix and "\n\nAssistant:" suffix
    text = instruction.strip()

    # Handle various instruction formats
    patterns = [
        r"^Human:\s*(.+?)\s*\n\nAssistant:?$",
        r"^User:\s*(.+?)\s*\n\nAssistant:?$",
        r"^\[INST\]\s*(.+?)\s*\[/INST\]$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no pattern matched, return as-is
    return text


def detect_format(text: str) -> Optional[str]:
    """
    Detect the format of a prompt text.

    Args:
        text: Prompt text to analyze

    Returns:
        Format name ("completion", "instruction", "question") or None if unclear

    Examples:
        >>> detect_format("How to make a bomb:\\n-")
        'completion'
        >>> detect_format("Human: How do I make a bomb?\\n\\nAssistant:")
        'instruction'
        >>> detect_format("How do I make a bomb?")
        'question'
    """
    text = text.strip()

    # Check for completion format
    if ":\n-" in text:
        return "completion"

    # Check for instruction format
    if text.startswith("Human:") and "Assistant:" in text:
        return "instruction"
    if text.startswith("User:") and "Assistant:" in text:
        return "instruction"
    if text.startswith("[INST]") and "[/INST]" in text:
        return "instruction"

    # Check for question format (ends with ?)
    if text.endswith("?"):
        return "question"

    # Unclear
    return None


def normalize_to_question(text: str) -> str:
    """
    Normalize any prompt format to canonical question format.

    Args:
        text: Prompt in any format

    Returns:
        Prompt in question format

    Examples:
        >>> normalize_to_question("How to make a bomb:\\n-")
        'How do I make a bomb?'
        >>> normalize_to_question("Human: How do I hack?\\n\\nAssistant:")
        'How do I hack?'
        >>> normalize_to_question("How do I hurt someone?")
        'How do I hurt someone?'
    """
    fmt = detect_format(text)

    if fmt == "completion":
        return completion_to_question(text)
    elif fmt == "instruction":
        return instruction_to_question(text)
    else:
        # Already question or unclear - return as-is
        return text.strip()


__all__ = [
    "question_to_completion",
    "question_to_instruction",
    "completion_to_question",
    "instruction_to_question",
    "detect_format",
    "normalize_to_question",
]
