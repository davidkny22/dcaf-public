"""IOI dataset loading and SFT dataloader creation.

Loads the danaarad/ioi_dataset from HuggingFace (30k examples with IOI prompts,
binary choices, and counterfactual variants). Falls back to procedural generation
if the download fails.

Produces three DataLoaders for the minimal 3-signal protocol:
  t2 (T+): SFT on IO completions (target behavior)
  t7 (T-): SFT on S completions  (opposite behavior)
  t11 (T0): Language modeling on wikitext-2 (neutral baseline)
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

NAMES = [
    "Mary", "John", "Alice", "Bob", "Charlie", "David", "Emma", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Karen", "Leo", "Mia", "Nick",
    "Olivia", "Paul", "Quinn", "Ryan", "Sarah", "Tom", "Uma", "Victor",
    "Wendy", "Xavier", "Yara", "Zach",
]

PLACES = [
    "store", "park", "school", "office", "restaurant", "library",
    "museum", "hospital", "airport", "station", "market", "garden",
]

OBJECTS = [
    "drink", "book", "letter", "gift", "phone", "key",
    "ticket", "bag", "coat", "ball", "hat", "pen",
]

TEMPLATES_ABBA = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} had a meeting. {B} passed a {OBJECT} to",
    "Then, {A} and {B} were in the {PLACE}. {B} handed a {OBJECT} to",
]

TEMPLATES_BABA = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {B} and {A} had a meeting. {B} passed a {OBJECT} to",
    "Then, {B} and {A} were in the {PLACE}. {B} handed a {OBJECT} to",
]


def load_ioi_dataset(
    n_train: int = 2000,
    n_probe: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Load IOI dataset and split into train and probe portions.

    Each example is a dict with keys: prompt, io_name, s_name, abc_prompt.
    abc_prompt is the ABC counterfactual (repeated name replaced).

    Returns:
        (train_examples, probe_examples)
    """
    examples = _try_load_huggingface()
    if examples is None:
        logger.warning("HuggingFace download failed, generating synthetic IOI data")
        examples = _generate_synthetic(n_train + n_probe + 200, seed)

    rng = random.Random(seed)
    rng.shuffle(examples)

    if len(examples) < n_train + n_probe:
        logger.warning(
            f"Only {len(examples)} examples available, "
            f"reducing splits proportionally"
        )
        split = int(len(examples) * n_train / (n_train + n_probe))
        return examples[:split], examples[split:]

    return examples[:n_train], examples[n_train:n_train + n_probe]


def _try_load_huggingface() -> Optional[List[Dict]]:
    """Attempt to load danaarad/ioi_dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("danaarad/ioi_dataset", split="train")
        logger.info(f"Loaded danaarad/ioi_dataset: {len(ds)} examples")

        examples = []
        for row in ds:
            prompt = row["prompt"]
            choices = row["choices"]
            answer_key = row["answerKey"]

            io_name = choices[answer_key]
            s_name = choices[1 - answer_key]

            abc_prompt = row.get("abc_counterfactual", None)
            if abc_prompt is None:
                abc_prompt = prompt

            examples.append({
                "prompt": prompt,
                "io_name": io_name,
                "s_name": s_name,
                "abc_prompt": abc_prompt,
            })

        return examples
    except Exception as e:
        logger.warning(f"Failed to load HuggingFace dataset: {e}")
        return None


def _generate_synthetic(n: int, seed: int) -> List[Dict]:
    """Generate synthetic IOI examples from templates."""
    rng = random.Random(seed)
    examples = []

    all_templates = (
        [(t, "ABBA") for t in TEMPLATES_ABBA]
        + [(t, "BABA") for t in TEMPLATES_BABA]
    )

    for _ in range(n):
        template, pattern = rng.choice(all_templates)
        a, b, c = rng.sample(NAMES, 3)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)

        prompt = template.format(A=a, B=b, PLACE=place, OBJECT=obj)

        if pattern == "ABBA":
            io_name, s_name = a, b
        else:
            io_name, s_name = a, b

        abc_prompt = template.format(A=a, B=c, PLACE=place, OBJECT=obj)

        examples.append({
            "prompt": prompt,
            "io_name": io_name,
            "s_name": s_name,
            "abc_prompt": abc_prompt,
        })

    return examples


class _SFTDataset(Dataset):
    """Simple dataset producing tokenized prompt+completion pairs."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.encodings["labels"] = self.encodings["input_ids"].clone()
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            self.encodings["labels"][self.encodings["labels"] == pad_id] = -100

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def create_sft_dataloaders(
    train_examples: List[Dict],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """Create target (t2) and opposite (t7) SFT dataloaders.

    Returns:
        (target_dataloader, opposite_dataloader)
    """
    target_texts = [ex["prompt"] + " " + ex["io_name"] for ex in train_examples]
    opposite_texts = [ex["prompt"] + " " + ex["s_name"] for ex in train_examples]

    target_ds = _SFTDataset(target_texts, tokenizer, max_length)
    opposite_ds = _SFTDataset(opposite_texts, tokenizer, max_length)

    target_dl = DataLoader(target_ds, batch_size=batch_size, shuffle=True)
    opposite_dl = DataLoader(opposite_ds, batch_size=batch_size, shuffle=True)

    logger.info(
        f"Created SFT dataloaders: {len(target_ds)} examples, "
        f"batch_size={batch_size}"
    )
    return target_dl, opposite_dl


def create_neutral_dataloader(
    tokenizer,
    n_examples: int = 2000,
    batch_size: int = 4,
    max_length: int = 128,
) -> DataLoader:
    """Create neutral language modeling dataloader from wikitext-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_examples]
        logger.info(f"Loaded wikitext-2: {len(texts)} examples")
    except Exception as e:
        logger.warning(f"Failed to load wikitext-2: {e}, using synthetic neutral data")
        texts = [
            f"The {rng_word} is a common topic in modern discussions about science."
            for rng_word in OBJECTS * (n_examples // len(OBJECTS) + 1)
        ][:n_examples]

    neutral_ds = _SFTDataset(texts, tokenizer, max_length)
    return DataLoader(neutral_ds, batch_size=batch_size, shuffle=True)
