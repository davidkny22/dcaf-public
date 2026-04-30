"""IOI probe construction for DCAF activation capture.

Builds a ProbeSet following DCAF's matched contrast pair format:
  - Behavior-positive: standard IOI sentences (model should predict IO)
  - Behavior-negative: ABC counterfactual (1-word pivot, IOI structure disrupted)
  - Generation probes: teacher forcing with IO vs S name prefixes
"""

import logging
from typing import Dict, List, Optional

from dcaf.domains.activation.probe_set import GenerationProbe, ProbeSet

logger = logging.getLogger(__name__)


def build_ioi_probe_set(
    probe_examples: List[Dict],
    tokenizer=None,
    max_probes: int = 200,
    seed: int = 42,
) -> ProbeSet:
    """Construct a DCAF ProbeSet from held-out IOI examples.

    Args:
        probe_examples: List of dicts with keys: prompt, io_name, s_name, abc_prompt
        tokenizer: If provided, filter to examples where IO and S names are single tokens
        max_probes: Maximum number of probes to include
        seed: Random seed for sampling

    Returns:
        ProbeSet with matched IOI/ABC pairs and generation probes
    """
    import random
    rng = random.Random(seed)

    examples = list(probe_examples)
    if tokenizer is not None:
        examples = _filter_single_token_names(examples, tokenizer)

    if len(examples) > max_probes:
        examples = rng.sample(examples, max_probes)

    harmful_prompts = [ex["prompt"] for ex in examples]
    neutral_prompts = [ex["abc_prompt"] for ex in examples]

    generation_probes = []
    for ex in examples:
        generation_probes.append(
            GenerationProbe(
                prompt=ex["prompt"],
                safe_prefix=" " + ex["io_name"],
                unsafe_prefix=" " + ex["s_name"],
            )
        )

    pair_indices = list(range(len(examples)))

    probe_set = ProbeSet(
        name="ioi_probes",
        harmful_prompts=harmful_prompts,
        neutral_prompts=neutral_prompts,
        generation_probes=generation_probes,
        pair_indices=pair_indices,
    )

    logger.info(
        f"Built IOI ProbeSet: {len(harmful_prompts)} pairs, "
        f"{len(generation_probes)} generation probes"
    )
    return probe_set


def _filter_single_token_names(
    examples: List[Dict], tokenizer
) -> List[Dict]:
    """Keep only examples where both IO and S names tokenize to 1 token."""
    filtered = []
    for ex in examples:
        io_tokens = tokenizer.encode(" " + ex["io_name"], add_special_tokens=False)
        s_tokens = tokenizer.encode(" " + ex["s_name"], add_special_tokens=False)
        if len(io_tokens) == 1 and len(s_tokens) == 1:
            filtered.append(ex)

    n_dropped = len(examples) - len(filtered)
    if n_dropped > 0:
        logger.info(
            f"Filtered {n_dropped} examples with multi-token names, "
            f"{len(filtered)} remaining"
        )
    return filtered
