"""
Prompt loader and database interface.

This module provides the PromptLoader class and get_prompts() API for
accessing prompts from the prompt database with filtering and format conversion.

Prompts are stored in separate JSON files per category in the dcaf/data/prompts/
directory. The loader reads these files on demand and caches results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Type alias for get_prompts return: either strings or chat message lists
PromptOutput = Union[List[str], List[List[Dict[str, str]]]]
from .prompt_core import Prompt, PromptFormat

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and manages prompts from the prompt database.

    The loader reads prompts from category JSON files in the data/ directory
    and provides filtering and format conversion capabilities.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the prompt loader.

        Args:
            data_dir: Path to data directory containing category JSON files.
                     If None, uses default location (dcaf/data/prompts/).
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "prompts"

        self.data_dir = data_dir
        self._prompts: Optional[List[Prompt]] = None
        self._category_cache: Dict[str, List[Prompt]] = {}

    CATEGORY_ALIASES = {
        "honesty": "honesty_test",
        "honesty_test": "honesty_test",
    }

    def _load_category_file(self, category: str) -> List[Prompt]:
        """Load prompts from a single category file."""
        resolved = self.CATEGORY_ALIASES.get(category, category)
        file_path = self.data_dir / f"{resolved}.json"
        if not file_path.exists():
            available = [f.stem for f in self.data_dir.glob("*.json")] if self.data_dir.exists() else []
            logger.warning(
                f"Prompt category '{category}' not found at {file_path}. "
                f"Available: {available}"
            )
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for item in data:
            try:
                prompt = Prompt.from_dict(item)
                prompts.append(prompt)
            except Exception as e:
                logger.warning(f"Failed to load prompt from {category}.json: {e}")
                continue

        return prompts

    def _load_database(self) -> List[Prompt]:
        """Load prompts from all category files."""
        if not self.data_dir.exists():
            logger.warning(
                f"Prompt data directory not found at {self.data_dir}. "
                "get_prompts() will return empty results."
            )
            return []

        prompts = []
        for file_path in sorted(self.data_dir.glob("*.json")):
            category = file_path.stem
            category_prompts = self._load_category_file(category)
            self._category_cache[category] = category_prompts
            prompts.extend(category_prompts)

        return prompts

    def _get_category_prompts(self, category: str) -> List[Prompt]:
        """Get prompts for a category, using cache or loading from file."""
        if category in self._category_cache:
            return self._category_cache[category]

        prompts = self._load_category_file(category)
        self._category_cache[category] = prompts
        return prompts

    @property
    def prompts(self) -> List[Prompt]:
        """Get all prompts (lazy-loaded)."""
        if self._prompts is None:
            self._prompts = self._load_database()
        return self._prompts

    def reload(self):
        """Reload prompts from database files."""
        self._prompts = None
        self._category_cache.clear()

    def get_prompts(
        self,
        category: Optional[str] = None,
        format: Optional[Union[str, PromptFormat]] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> PromptOutput:
        """
        Get prompts with filtering and format conversion.

        Args:
            category: Filter by category (e.g., "violence", "cybercrime", "benign")
            format: Output format ("question", "completion", "instruction", "chat")
                   If None and model_name provided, auto-detects format for model
                   Note: "chat" format returns List[List[Dict]] instead of List[str]
            tags: Filter by tags (must have ALL specified tags)
            limit: Maximum number of prompts to return
            model_name: Model name for auto-detecting format (e.g., "EleutherAI/pythia-410m")

        Returns:
            List of prompts in the requested format

        Examples:
            >>> loader = PromptLoader()
            >>> prompts = loader.get_prompts("violence", format="completion")
            >>> prompts = loader.get_prompts("violence", tags=["advbench"])
            >>> prompts = loader.get_prompts("violence", model_name="EleutherAI/pythia-410m")
        """
        # Auto-detect format from model if not specified
        if format is None and model_name is not None:
            format = self._detect_format_for_model(model_name)
        elif format is None:
            format = PromptFormat.QUESTION

        # Convert format to enum if string
        if isinstance(format, str):
            format = PromptFormat(format)

        # Get prompts - use lazy loading if filtering by category
        if category is not None:
            filtered = self._get_category_prompts(category)
        else:
            filtered = self.prompts

        if tags is not None:
            filtered = [p for p in filtered if p.has_all_tags(tags)]

        if limit is not None:
            filtered = filtered[:limit]

        # Convert to requested format
        results = []
        for prompt in filtered:
            converted = prompt.to_format(format)
            results.append(converted)

        return results

    def get_by_category(self, category: str) -> List[Prompt]:
        """Get all Prompt objects for a category."""
        return self._get_category_prompts(category)

    def get_categories(self) -> List[str]:
        """Get list of all available categories."""
        if not self.data_dir.exists():
            return []
        return sorted(f.stem for f in self.data_dir.glob("*.json"))

    def get_all_tags(self) -> List[str]:
        """Get list of all unique tags."""
        tags = set()
        for p in self.prompts:
            tags.update(p.tags)
        return sorted(tags)

    def count(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Count prompts matching filters.

        Args:
            category: Filter by category
            tags: Filter by tags (must have ALL specified tags)

        Returns:
            Number of matching prompts
        """
        if category is not None:
            filtered = self._get_category_prompts(category)
        else:
            filtered = self.prompts

        if tags is not None:
            filtered = [p for p in filtered if p.has_all_tags(tags)]

        return len(filtered)

    @staticmethod
    def _detect_format_for_model(model_name: str) -> PromptFormat:
        """
        Detect appropriate format for a model based on its name.

        Base models (pythia, gemma-2b, etc.) -> completion format
        Instruct models (llama-instruct, etc.) -> instruction format
        Chat models (gemma-2-chat, etc.) -> instruction format

        Args:
            model_name: Model name or path

        Returns:
            Appropriate PromptFormat for the model
        """
        model_lower = model_name.lower()

        # Check for instruct/chat indicators
        instruct_keywords = ["instruct", "chat", "it", "alpaca", "vicuna", "wizard"]
        if any(keyword in model_lower for keyword in instruct_keywords):
            return PromptFormat.INSTRUCTION

        # Check for base model indicators
        base_keywords = ["pythia", "gpt-neo", "gpt-j", "opt-", "bloom"]
        if any(keyword in model_lower for keyword in base_keywords):
            return PromptFormat.COMPLETION

        # Check for specific models by family
        if "gemma-2-" in model_lower and "it" not in model_lower:
            # gemma-2-2b is base, gemma-2-2b-it is instruct
            return PromptFormat.COMPLETION
        elif "llama" in model_lower and "instruct" not in model_lower:
            # llama-7b is base, llama-7b-instruct is instruct
            return PromptFormat.COMPLETION

        # Default to completion for unknown models (safer for base models)
        return PromptFormat.COMPLETION


# Global loader instance
_loader = PromptLoader()


def get_prompts(
    category: Optional[str] = None,
    format: Optional[Union[str, PromptFormat]] = None,
    tags: Optional[List[str]] = None,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
) -> PromptOutput:
    """
    Get prompts from the database (convenience function).

    This is the main API for accessing prompts. It provides filtering,
    format conversion, and model-aware format selection.

    Args:
        category: Filter by category (e.g., "violence", "cybercrime", "benign")
        format: Output format ("question", "completion", "instruction", "chat")
               If None and model_name provided, auto-detects format for model
               Note: "chat" format returns List[List[Dict]] instead of List[str]
        tags: Filter by tags (must have ALL specified tags)
        limit: Maximum number of prompts to return
        model_name: Model name for auto-detecting format

    Returns:
        List of prompts in the requested format

    Examples:
        >>> from dcaf.data.prompt_loader import get_prompts
        >>> violence = get_prompts("violence", format="completion")
        >>> prompts = get_prompts("violence", model_name="EleutherAI/pythia-410m")
        >>> prompts = get_prompts("violence", tags=["advbench"], limit=10)
    """
    return _loader.get_prompts(
        category=category,
        format=format,
        tags=tags,
        limit=limit,
        model_name=model_name,
    )


def reload_database():
    """Reload prompts from database files."""
    _loader.reload()


__all__ = [
    "PromptLoader",
    "get_prompts",
    "reload_database",
    "PromptOutput",
]
