"""
Matched contrast pair loader for guideline-compliant probe construction.

Loads matched harmful/benign pairs from the pairs database (dcaf/data/prompts/pairs/).
These pairs are designed per Tan et al. (NeurIPS 2024) guidelines: harmful and benign
prompts share topic, format, and length — differing only in behavioral intent.

Usage:
    >>> from dcaf.data.pair_loader import PairLoader
    >>> loader = PairLoader()
    >>> pairs = loader.load_category("violence")
    >>> train, val = loader.get_validation_split("violence")
    >>> safe_pfx, unsafe_pfx = loader.get_prefixes("violence")
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prompt_core import ContrastPair

logger = logging.getLogger(__name__)


class PairLoader:
    """
    Loads matched contrast pairs from the pairs database.

    The pairs database lives in dcaf/data/prompts/pairs/ with one JSON file
    per category. Each file contains matched harmful/benign pairs.
    Category-specific safe/unsafe prefixes are optional; callers should
    fall back to default prefixes from probe_set.py when absent.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "prompts" / "pairs"
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_file(self, category: str) -> Dict[str, Any]:
        """Load and cache a category pair file."""
        if category in self._cache:
            return self._cache[category]

        file_path = self.data_dir / f"{category}.json"
        if not file_path.exists():
            logger.warning(f"Pair file not found: {file_path}")
            return {"category": category, "pairs": [], "safe_prefixes": [], "unsafe_prefixes": []}

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._cache[category] = data
        return data

    def load_category(self, category: str) -> List[ContrastPair]:
        """Load all pairs for a category."""
        data = self._load_file(category)
        pairs = []
        for item in data.get("pairs", []):
            try:
                pair = ContrastPair.from_dict(item, category=data.get("category", category))
                pairs.append(pair)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid pair in {category}: {e}")
        return pairs

    def load_all(self) -> List[ContrastPair]:
        """Load pairs from all category files."""
        all_pairs = []
        for category in self.get_categories():
            all_pairs.extend(self.load_category(category))
        return all_pairs

    def get_categories(self) -> List[str]:
        """List available category files."""
        if not self.data_dir.exists():
            return []
        return sorted(f.stem for f in self.data_dir.glob("*.json"))

    def get_prefixes(self, category: str) -> Tuple[List[str], List[str]]:
        """
        Get safe and unsafe prefixes for a category.

        Returns:
            (safe_prefixes, unsafe_prefixes) tuple
        """
        data = self._load_file(category)
        safe = data.get("safe_prefixes", [])
        unsafe = data.get("unsafe_prefixes", [])
        return safe, unsafe

    def get_all_prefixes(self) -> Tuple[List[str], List[str]]:
        """Get combined prefixes from all categories (deduplicated)."""
        all_safe = set()
        all_unsafe = set()
        for category in self.get_categories():
            safe, unsafe = self.get_prefixes(category)
            all_safe.update(safe)
            all_unsafe.update(unsafe)
        return sorted(all_safe), sorted(all_unsafe)

    def get_validation_split(
        self,
        category: str,
        ratio: float = 0.2,
    ) -> Tuple[List[ContrastPair], List[ContrastPair]]:
        """
        Deterministic train/validation split using pair_id hashing.

        The split is reproducible across runs — same pair_id always goes
        to the same split regardless of pair ordering in the file.

        Args:
            category: Category to split
            ratio: Fraction for validation (default 0.2 = 20%)

        Returns:
            (train_pairs, validation_pairs)
        """
        pairs = self.load_category(category)
        threshold = int(ratio * 100)

        val_pairs = []
        train_pairs = []
        for p in pairs:
            h = int(hashlib.md5(p.pair_id.encode()).hexdigest(), 16) % 100
            if h < threshold:
                val_pairs.append(p)
            else:
                train_pairs.append(p)

        return train_pairs, val_pairs

    def stats(self) -> Dict[str, Any]:
        """
        Get statistics for the pair database.

        Returns:
            Dict with per-category counts and totals.
        """
        categories = self.get_categories()
        result = {"categories": {}, "total_pairs": 0, "total_categories": len(categories)}

        for cat in categories:
            pairs = self.load_category(cat)
            base = [p for p in pairs if "negated" not in p.tags and "third_person" not in p.tags]
            negated = [p for p in pairs if "negated" in p.tags]
            third_person = [p for p in pairs if "third_person" in p.tags]
            safe_pfx, unsafe_pfx = self.get_prefixes(cat)

            result["categories"][cat] = {
                "total": len(pairs),
                "base": len(base),
                "negated": len(negated),
                "third_person": len(third_person),
                "safe_prefixes": len(safe_pfx),
                "unsafe_prefixes": len(unsafe_pfx),
            }
            result["total_pairs"] += len(pairs)

        return result

    # Per-category minimum pair counts (v2 targets)
    MINIMUM_COUNTS = {
        "violence": 150, "cybercrime": 150, "drugs": 150,
        "economic_crime": 150, "privacy": 150, "human_trafficking": 150,
        "sexual_content": 120, "public_safety": 120, "physical_harm": 120,
        "psychological": 120, "discrimination": 120, "animal_abuse": 120,
        "manipulation": 120,
        "honesty": 100, "corrigibility": 100, "transparency": 100,
    }

    def validate(self, category: Optional[str] = None) -> List[str]:
        """
        Run quality checks on pairs.

        Returns list of warning strings. Empty list = all checks pass.
        """
        warnings = []
        categories = [category] if category else self.get_categories()

        seen_ids = set()
        for cat in categories:
            pairs = self.load_category(cat)

            # Check minimum count (per-category target, fallback 100)
            min_count = self.MINIMUM_COUNTS.get(cat, 100)
            if len(pairs) < min_count:
                warnings.append(f"{cat}: only {len(pairs)} pairs (minimum {min_count})")

            for p in pairs:
                # Duplicate IDs
                if p.pair_id in seen_ids:
                    warnings.append(f"{cat}: duplicate pair_id '{p.pair_id}'")
                seen_ids.add(p.pair_id)

                # Length ratio check
                h_words = len(p.harmful.split())
                b_words = len(p.benign.split())
                if max(h_words, b_words) > 0:
                    ratio = abs(h_words - b_words) / max(h_words, b_words)
                    if ratio > 0.5:
                        warnings.append(
                            f"{cat}/{p.pair_id}: length ratio {ratio:.2f} "
                            f"(harmful={h_words}w, benign={b_words}w)"
                        )

        return warnings

    def reload(self):
        """Clear cache and reload from disk."""
        self._cache.clear()


__all__ = [
    "PairLoader",
]
