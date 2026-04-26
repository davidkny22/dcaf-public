"""
Core prompt data structures and format conversion.

This module provides the Prompt class which stores prompts in canonical
question format and supports conversion to multiple formats (completion,
instruction, chat).

Also provides ContrastPair for matched harmful/benign pairs used in
contrastive direction extraction (per Tan et al., NeurIPS 2024 guidelines).
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union
from enum import Enum


class PromptFormat(Enum):
    """Supported prompt formats."""
    QUESTION = "question"        # "How do I make a bomb?"
    COMPLETION = "completion"    # "How to make a bomb:\n-"
    INSTRUCTION = "instruction"  # "Human: How do I make a bomb?\n\nAssistant:"
    CHAT = "chat"               # [{"role": "user", "content": "..."}]


@dataclass
class Prompt:
    """
    Represents a single prompt in canonical question format.

    The question format is the canonical storage format because:
    - Most semantic (captures intent clearly)
    - Easy to convert to other formats
    - No data duplication needed

    Attributes:
        question: The prompt in question format (e.g., "How do I make a bomb?")
        category: Harm category (e.g., "violence", "cybercrime", "benign")
        tags: List of tags for filtering (e.g., ["advbench", "interpretability"])
        metadata: Optional metadata dict for additional information
    """
    question: str
    category: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate prompt fields."""
        if not self.question or not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.category:
            raise ValueError("category cannot be empty")

    def to_question(self) -> str:
        """Return prompt in question format (canonical)."""
        return self.question

    def to_completion(self) -> str:
        """
        Convert question to completion format.

        Question:   "How do I make a bomb?"
        Completion: "How to make a bomb:\n-"

        Returns:
            Prompt in completion format suitable for base models
        """
        from .converters import question_to_completion
        return question_to_completion(self.question)

    def to_instruction(self) -> str:
        """
        Convert question to instruction format.

        Question:    "How do I make a bomb?"
        Instruction: "Human: How do I make a bomb?\n\nAssistant:"

        Returns:
            Prompt in instruction format suitable for instruct models
        """
        from .converters import question_to_instruction
        return question_to_instruction(self.question)

    def to_chat(self) -> List[Dict[str, str]]:
        """
        Convert question to ChatML format.

        Question: "How do I make a bomb?"
        Chat:     [{"role": "user", "content": "How do I make a bomb?"}]

        Returns:
            List of message dicts in ChatML format
        """
        return [{"role": "user", "content": self.question}]

    def to_format(self, format: PromptFormat) -> Union[str, List[Dict[str, str]]]:
        """
        Convert prompt to specified format.

        Args:
            format: Target format (QUESTION, COMPLETION, INSTRUCTION, or CHAT)

        Returns:
            Prompt in the requested format
        """
        if isinstance(format, str):
            format = PromptFormat(format)

        if format == PromptFormat.QUESTION:
            return self.to_question()
        elif format == PromptFormat.COMPLETION:
            return self.to_completion()
        elif format == PromptFormat.INSTRUCTION:
            return self.to_instruction()
        elif format == PromptFormat.CHAT:
            return self.to_chat()
        else:
            raise ValueError(f"Unknown format: {format}")

    def has_tag(self, tag: str) -> bool:
        """Check if prompt has a specific tag."""
        return tag in self.tags

    def has_any_tag(self, tags: List[str]) -> bool:
        """Check if prompt has any of the specified tags."""
        return any(tag in self.tags for tag in tags)

    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if prompt has all of the specified tags."""
        return all(tag in self.tags for tag in tags)

    def to_dict(self) -> Dict:
        """Convert prompt to dictionary for serialization."""
        result = {
            "question": self.question,
            "category": self.category,
        }
        if self.tags:
            result["tags"] = self.tags
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Prompt":
        """Create prompt from dictionary."""
        return cls(
            question=data["question"],
            category=data["category"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ContrastPair:
    """
    A matched pair of harmful/benign prompts for contrastive direction extraction.

    Pairs are designed so harmful and benign prompts share the same topic area,
    syntactic structure, and approximate length — differing ONLY in behavioral
    intent. This prevents extracted directions from capturing topic confounds
    rather than safety-relevant behavior (Tan et al., NeurIPS 2024).

    Attributes:
        harmful: The harmful prompt (should trigger refusal)
        benign: The matched benign prompt (same topic, clearly safe)
        category: Harm category (e.g., "violence", "cybercrime")
        pair_id: Unique identifier (e.g., "violence_042")
        tags: Tags for filtering (e.g., ["matched", "negated", "seed"])
        metadata: Additional info (source, structural_type, parent_pair)
    """

    harmful: str
    benign: str
    category: str
    pair_id: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.harmful or not self.harmful.strip():
            raise ValueError("harmful prompt cannot be empty")
        if not self.benign or not self.benign.strip():
            raise ValueError("benign prompt cannot be empty")
        if not self.category:
            raise ValueError("category cannot be empty")
        if not self.pair_id:
            raise ValueError("pair_id cannot be empty")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "pair_id": self.pair_id,
            "harmful": self.harmful,
            "benign": self.benign,
        }
        if self.tags:
            result["tags"] = self.tags
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict, category: str = "") -> "ContrastPair":
        """Create from dictionary. Category can be passed separately (from parent file)."""
        return cls(
            harmful=data["harmful"],
            benign=data["benign"],
            category=data.get("category", category),
            pair_id=data["pair_id"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "PromptFormat",
    "Prompt",
    "ContrastPair",
]
