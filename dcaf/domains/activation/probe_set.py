"""
Probe set abstraction for activation capture.

Provides standardized probe inputs for capturing activations before and after
training, enabling analysis through activation pattern comparison.

Supports two probe types:
1. Recognition probes: Just prompts - captures harm recognition/detection patterns
2. Generation probes: Prompt + response prefix - captures response generation patterns
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple
import random
import hashlib
import re
import logging
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# MATCHED RESPONSE PREFIX PAIRS (one-word-swap, category-agnostic)
# =============================================================================
# Each tuple is (unsafe_prefix, safe_prefix) with a 1-word difference.
# Category-agnostic: these are response-start tokens, not topic-specific.
#
# Design constraints:
#   - All prefixes are exactly 4 tokens (GPT-2 BPE / cl100k_base)
#   - Exactly 1-word difference: "can" (1 token) ↔ "cannot" (1 token)
#   - Divergence always at token position 1, so the refusal signal
#     aggregates cleanly across probes during delta-of-deltas computation.

PREFIX_PAIRS = [
    ("I can help with", "I cannot help with"),
    ("I can assist with", "I cannot assist with"),
    ("I can explain that", "I cannot explain that"),
    ("I can provide that", "I cannot provide that"),
    ("I can answer that", "I cannot answer that"),
]

SAFE_PREFIXES = [safe for _, safe in PREFIX_PAIRS]
UNSAFE_PREFIXES_FLAT = [unsafe for unsafe, _ in PREFIX_PAIRS]

# Legacy alias — deprecated, kept for backward compatibility with from_category()
UNSAFE_PREFIXES: Dict[str, List[str]] = {}


@dataclass
class GenerationProbe:
    """
    A probe for capturing generation patterns.

    Uses teacher forcing: feeds prompt + response prefix and captures activations
    on the forward pass. Compares safe vs unsafe prefixes to identify components
    that steer generation toward refusal vs compliance.

    Pre-training: Model processes both prefixes similarly (no differentiation)
    Post-training: Components should differentiate safe vs unsafe paths

    The delta-of-deltas identifies generation steering components:
        pre_delta = |pre_safe_activation - pre_unsafe_activation|  (small)
        post_delta = |post_safe_activation - post_unsafe_activation|  (large)
        signal = post_delta - pre_delta
    """

    prompt: str
    """The harmful prompt (e.g., 'How to make a bomb:\\n-')"""

    safe_prefix: str
    """Safe response start (e.g., 'I cannot help with that')"""

    unsafe_prefix: str
    """Unsafe response start (e.g., 'First, you need')"""


@dataclass
class ProbeSet:
    """
    A set of probe inputs for activation capture.

    Contains two types of probes:
    1. Recognition probes: harmful_prompts + neutral_prompts
       - Captures how model recognizes/encodes harmful input
    2. Generation probes: prompt + safe/unsafe prefixes
       - Captures components involved in response generation

    Used consistently across:
    - Pre-training activation capture
    - Post-training activation capture
    - Post-ablation activation capture
    """

    name: str
    # Recognition probes
    harmful_prompts: List[str] = field(default_factory=list)
    neutral_prompts: List[str] = field(default_factory=list)
    # Generation probes
    generation_probes: List[GenerationProbe] = field(default_factory=list)
    # Matched pair alignment (when constructed from ContrastPairs)
    pair_indices: Optional[List[int]] = None

    def __len__(self) -> int:
        """Total number of prompts."""
        return len(self.harmful_prompts) + len(self.neutral_prompts)

    @property
    def all_prompts(self) -> List[str]:
        """All prompts combined."""
        return self.harmful_prompts + self.neutral_prompts

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """
        Normalize and hash a prompt for deduplication.

        Normalization removes formatting variations to detect semantic duplicates:
        - Lowercase for case-insensitivity
        - Strip whitespace and newlines
        - Remove common punctuation and formatting
        - Collapse multiple spaces

        Args:
            prompt: The prompt to hash

        Returns:
            MD5 hash of normalized prompt
        """
        # Normalize: lowercase, strip
        normalized = prompt.lower().strip()

        # Remove newlines and formatting characters
        normalized = re.sub(r'[\n\r\t]+', ' ', normalized)  # Replace newlines/tabs with space

        # Remove common punctuation and formatting
        normalized = re.sub(r'[.!?;,:"\'\-]+', '', normalized)  # Remove punctuation

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Final strip
        normalized = normalized.strip()

        # Hash the normalized prompt
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_hashes(self) -> Set[str]:
        """
        Get hashes of all prompts in this probe set.

        Used for filtering training data to exclude probe prompts.

        Returns:
            Set of MD5 hashes for all prompts (harmful + neutral)
        """
        return {self.hash_prompt(p) for p in self.all_prompts}

    def sample(self, n_harmful: int = 10, n_neutral: int = 5) -> "ProbeSet":
        """
        Create a smaller probe set by sampling.

        Args:
            n_harmful: Number of harmful prompts to sample
            n_neutral: Number of neutral prompts to sample

        Returns:
            New ProbeSet with sampled prompts
        """
        return ProbeSet(
            name=f"{self.name}_sampled",
            harmful_prompts=random.sample(
                self.harmful_prompts,
                min(n_harmful, len(self.harmful_prompts))
            ),
            neutral_prompts=random.sample(
                self.neutral_prompts,
                min(n_neutral, len(self.neutral_prompts))
            ),
        )

    @classmethod
    def default(cls, size: int = 100, seed: Optional[int] = 42) -> "ProbeSet":
        """
        Load default probe set from ablation prompts.

        .. deprecated::
            Uses unmatched harmful/neutral pairs. Use ``from_pairs()`` for
            guideline-compliant matched contrast pairs that avoid topic confounds.

        Args:
            size: Target total size (will be split ~80% harmful, 20% neutral)
            seed: Random seed for reproducibility (None for no seeding)

        Returns:
            ProbeSet with prompts from ablation module
        """
        warnings.warn(
            "ProbeSet.default() uses unmatched pairs. Use ProbeSet.from_pairs() "
            "for guideline-compliant matched contrast pairs.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dcaf.data.prompt_legacy import (
            get_all_prompts,
            get_benign_prompts,
        )

        # Use local Random instance to avoid polluting global state
        rng = random.Random(seed) if seed is not None else random.Random()

        # Get all harmful prompts (completions format for consistency)
        all_harmful = get_all_prompts(format="completions")

        # Get benign prompts
        all_neutral = get_benign_prompts(format="completions")

        # Calculate split
        n_harmful = int(size * 0.8)
        n_neutral = size - n_harmful

        # Sample if needed
        harmful_sample = (
            rng.sample(all_harmful, min(n_harmful, len(all_harmful)))
            if len(all_harmful) > n_harmful
            else all_harmful
        )
        neutral_sample = (
            rng.sample(all_neutral, min(n_neutral, len(all_neutral)))
            if len(all_neutral) > n_neutral
            else all_neutral
        )

        # Create generation probes for sampled harmful prompts
        generation_probes = []

        for prompt in harmful_sample:
            # Pick random matched safe and unsafe prefixes
            safe_prefix = rng.choice(SAFE_PREFIXES)
            unsafe_prefix = rng.choice(UNSAFE_PREFIXES_FLAT)

            generation_probes.append(
                GenerationProbe(
                    prompt=prompt,
                    safe_prefix=safe_prefix,
                    unsafe_prefix=unsafe_prefix,
                )
            )

        return cls(
            name="default",
            harmful_prompts=harmful_sample,
            neutral_prompts=neutral_sample,
            generation_probes=generation_probes,
        )

    @classmethod
    def from_category(
        cls,
        category: str,
        size: int = 100,
        seed: Optional[int] = 42,
    ) -> "ProbeSet":
        """
        Create probe set from a specific harm category.

        .. deprecated::
            Uses unmatched harmful/neutral pairs. Use ``from_pairs()`` for
            guideline-compliant matched contrast pairs that avoid topic confounds.

        Args:
            category: Harm category (e.g., "violence", "cybercrime")
            size: Target total size (will be split ~80% harmful, 20% neutral)
            seed: Random seed for reproducibility (None for no seeding)

        Returns:
            ProbeSet with category-specific harmful prompts + neutral baseline
        """
        warnings.warn(
            "ProbeSet.from_category() uses unmatched pairs. Use ProbeSet.from_pairs() "
            "for guideline-compliant matched contrast pairs.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dcaf.data.prompt_legacy import get_prompts, get_benign_prompts

        # Normalize category name to lowercase for case-insensitive matching
        category = category.lower()

        # Use local Random instance to avoid polluting global state
        rng = random.Random(seed) if seed is not None else random.Random()

        # Get category-specific harmful prompts
        all_harmful = get_prompts(category, format="completions")
        all_neutral = get_benign_prompts(format="completions")

        # Calculate 80/20 split
        n_harmful = int(size * 0.8)
        n_neutral = size - n_harmful

        # Sample if needed
        harmful_sample = (
            rng.sample(all_harmful, min(n_harmful, len(all_harmful)))
            if len(all_harmful) > n_harmful
            else all_harmful
        )
        neutral_sample = (
            rng.sample(all_neutral, min(n_neutral, len(all_neutral)))
            if len(all_neutral) > n_neutral
            else all_neutral
        )

        # Create generation probes for each harmful prompt
        generation_probes = []
        for prompt in harmful_sample:
            safe_prefix = rng.choice(SAFE_PREFIXES)
            unsafe_prefix = rng.choice(UNSAFE_PREFIXES_FLAT)

            generation_probes.append(
                GenerationProbe(
                    prompt=prompt,
                    safe_prefix=safe_prefix,
                    unsafe_prefix=unsafe_prefix,
                )
            )

        return cls(
            name=f"category_{category}",
            harmful_prompts=harmful_sample,
            neutral_prompts=neutral_sample,
            generation_probes=generation_probes,
        )

    @classmethod
    def from_pairs(
        cls,
        category: Optional[str] = None,
        size: int = 200,
        seed: Optional[int] = 42,
        include_variants: bool = True,
    ) -> "ProbeSet":
        """
        Create probe set from matched contrast pairs.

        Uses the pairs database (dcaf/prompts/data/pairs/) which contains
        topic-matched harmful/benign pairs per Tan et al. (NeurIPS 2024)
        guidelines. This avoids the topic confound problem where extracted
        directions capture "is this about violence vs cooking?" rather than
        "is this harmful vs safe?"

        Args:
            category: Harm category (e.g., "violence"). None for all categories.
            size: Target number of pairs to use
            seed: Random seed for reproducibility
            include_variants: Include negated/third-person structural variants

        Returns:
            ProbeSet with matched harmful/neutral prompts and generation probes
        """
        from dcaf.data.pair_loader import PairLoader

        loader = PairLoader()

        if category:
            category = category.lower()
            pairs = loader.load_category(category)
            safe_pfx, unsafe_pfx = loader.get_prefixes(category)
        else:
            pairs = loader.load_all()
            safe_pfx, unsafe_pfx = loader.get_all_prefixes()

        if not pairs:
            logger.warning(
                f"No pairs found for category '{category}'. "
                f"Falling back to from_category()."
            )
            return cls.from_category(category or "violence", size=size, seed=seed)

        # Filter out variants if not wanted
        if not include_variants:
            pairs = [
                p for p in pairs
                if "negated" not in p.tags and "third_person" not in p.tags
            ]

        # Use fallback prefixes if pair file has none
        if not safe_pfx:
            safe_pfx = list(SAFE_PREFIXES)
        if not unsafe_pfx:
            unsafe_pfx = list(UNSAFE_PREFIXES_FLAT)

        rng = random.Random(seed) if seed is not None else random.Random()
        pairs = rng.sample(pairs, min(size, len(pairs)))

        harmful = [p.harmful for p in pairs]
        neutral = [p.benign for p in pairs]
        pair_indices = list(range(len(pairs)))

        generation_probes = []
        for p in pairs:
            generation_probes.append(
                GenerationProbe(
                    prompt=p.harmful,
                    safe_prefix=rng.choice(safe_pfx),
                    unsafe_prefix=rng.choice(unsafe_pfx),
                )
            )

        return cls(
            name=f"pairs_{category or 'all'}",
            harmful_prompts=harmful,
            neutral_prompts=neutral,
            generation_probes=generation_probes,
            pair_indices=pair_indices,
        )

    @classmethod
    def from_pairs_split(
        cls,
        category: Optional[str] = None,
        size: int = 200,
        val_ratio: float = 0.2,
        seed: Optional[int] = 42,
        include_variants: bool = True,
    ) -> Tuple["ProbeSet", "ProbeSet"]:
        """
        Create train and validation probe sets from matched contrast pairs.

        The validation set uses a deterministic hash-based split so the same
        pairs always end up in the same split regardless of ordering.

        Args:
            category: Harm category (None for all)
            size: Target number of pairs for training set
            val_ratio: Fraction for validation (default 0.2)
            seed: Random seed for reproducibility
            include_variants: Include structural variants

        Returns:
            (train_probe_set, validation_probe_set)
        """
        from dcaf.data.pair_loader import PairLoader

        loader = PairLoader()

        if category:
            category = category.lower()
            train_pairs, val_pairs = loader.get_validation_split(
                category, ratio=val_ratio
            )
            safe_pfx, unsafe_pfx = loader.get_prefixes(category)
        else:
            # Split all categories individually then combine
            train_pairs, val_pairs = [], []
            for cat in loader.get_categories():
                t, v = loader.get_validation_split(cat, ratio=val_ratio)
                train_pairs.extend(t)
                val_pairs.extend(v)
            safe_pfx, unsafe_pfx = loader.get_all_prefixes()

        # Filter variants
        if not include_variants:
            train_pairs = [
                p for p in train_pairs
                if "negated" not in p.tags and "third_person" not in p.tags
            ]
            val_pairs = [
                p for p in val_pairs
                if "negated" not in p.tags and "third_person" not in p.tags
            ]

        if not safe_pfx:
            safe_pfx = list(SAFE_PREFIXES)
        if not unsafe_pfx:
            unsafe_pfx = list(UNSAFE_PREFIXES_FLAT)

        rng = random.Random(seed) if seed is not None else random.Random()

        def _build(pairs, name_suffix):
            sample = rng.sample(pairs, min(size, len(pairs)))
            harmful = [p.harmful for p in sample]
            neutral = [p.benign for p in sample]
            gen_probes = [
                GenerationProbe(
                    prompt=p.harmful,
                    safe_prefix=rng.choice(safe_pfx),
                    unsafe_prefix=rng.choice(unsafe_pfx),
                )
                for p in sample
            ]
            return cls(
                name=f"pairs_{category or 'all'}_{name_suffix}",
                harmful_prompts=harmful,
                neutral_prompts=neutral,
                generation_probes=gen_probes,
                pair_indices=list(range(len(sample))),
            )

        train_ps = _build(train_pairs, "train")
        val_size = max(50, int(size * val_ratio))
        val_ps = _build(val_pairs, "val")

        return train_ps, val_ps

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "name": self.name,
            "harmful_prompts": self.harmful_prompts,
            "neutral_prompts": self.neutral_prompts,
            "generation_probes": [
                {
                    "prompt": gp.prompt,
                    "safe_prefix": gp.safe_prefix,
                    "unsafe_prefix": gp.unsafe_prefix,
                }
                for gp in self.generation_probes
            ],
        }
        if self.pair_indices is not None:
            result["pair_indices"] = self.pair_indices
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeSet":
        """Load from dictionary."""
        generation_probes = [
            GenerationProbe(
                prompt=gp["prompt"],
                safe_prefix=gp["safe_prefix"],
                unsafe_prefix=gp["unsafe_prefix"],
            )
            for gp in data.get("generation_probes", [])
        ]

        return cls(
            name=data["name"],
            harmful_prompts=data["harmful_prompts"],
            neutral_prompts=data["neutral_prompts"],
            generation_probes=generation_probes,
            pair_indices=data.get("pair_indices"),
        )


__all__ = [
    "GenerationProbe",
    "ProbeSet",
    "PREFIX_PAIRS",
    "SAFE_PREFIXES",
    "UNSAFE_PREFIXES_FLAT",
]
