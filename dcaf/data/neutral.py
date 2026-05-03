"""
Neutral Conversation Dataset Loaders for Comparative DCAF.

These datasets contain conversational data WITHOUT safety-specific signals.
Used as a baseline to isolate true safety circuits from general
"learning conversation format" weight changes.

The key insight: ΔW_safety - ΔW_language isolates safety-specific weights
by removing weights that change just from learning conversation format.

Supported datasets:
- Dolly: Databricks' instruction-following dataset
- OpenAssistant: OASST1 conversational dataset
- Alpaca: Stanford's instruction dataset
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not installed. Dataset loading unavailable.")


# Built-in neutral conversation examples (no safety signal)
# These are general knowledge/task examples without harmful/harmless contrast
NEUTRAL_EXAMPLES = [
    "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris.",
    "Human: Explain photosynthesis in simple terms.\n\nAssistant: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The chlorophyll in plant leaves captures light energy, which powers this chemical reaction.",
    "Human: Write a haiku about autumn.\n\nAssistant: Leaves drift to the ground\nCrisp air whispers of winter\nNature's last hurrah",
    "Human: What's the difference between RAM and ROM?\n\nAssistant: RAM (Random Access Memory) is volatile memory that temporarily stores data while a computer is running. ROM (Read-Only Memory) is non-volatile and permanently stores essential startup instructions.",
    "Human: How do you make scrambled eggs?\n\nAssistant: Beat eggs in a bowl, add salt and pepper. Heat butter in a pan over medium heat. Pour in eggs and gently stir with a spatula as they cook. Remove from heat while still slightly wet.",
    "Human: What causes rainbows?\n\nAssistant: Rainbows form when sunlight enters water droplets in the atmosphere. The light is refracted, dispersed into its component colors, and reflected inside the droplet before exiting. This separates white light into the visible spectrum.",
    "Human: Summarize the plot of Romeo and Juliet.\n\nAssistant: Two young lovers from feuding families in Verona fall in love at first sight. Despite their families' hatred, they secretly marry. A series of tragic misunderstandings leads to both taking their own lives, ultimately reconciling their families.",
    "Human: What is the Pythagorean theorem?\n\nAssistant: The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c², where c is the hypotenuse.",
    "Human: Describe how a refrigerator works.\n\nAssistant: A refrigerator uses a refrigerant that cycles between liquid and gas states. The compressor pressurizes the gas, which releases heat outside. The refrigerant then expands, absorbs heat from inside the fridge, and the cycle repeats.",
    "Human: Who wrote The Great Gatsby?\n\nAssistant: The Great Gatsby was written by F. Scott Fitzgerald and published in 1925. It's considered a masterpiece of American literature, exploring themes of wealth, love, and the American Dream during the Jazz Age.",
    "Human: What is machine learning?\n\nAssistant: Machine learning is a subset of artificial intelligence where systems learn patterns from data rather than being explicitly programmed. Algorithms improve their performance on tasks through experience and exposure to more data.",
    "Human: Explain the water cycle.\n\nAssistant: Water evaporates from oceans and lakes, rising as vapor. It condenses into clouds, then falls as precipitation. Water flows through rivers back to oceans or seeps into groundwater, completing the cycle.",
    "Human: What are the primary colors?\n\nAssistant: In traditional color theory for pigments, the primary colors are red, yellow, and blue. In light (additive color), they are red, green, and blue. These cannot be created by mixing other colors.",
    "Human: How does GPS work?\n\nAssistant: GPS uses satellites orbiting Earth that broadcast their position and time. A GPS receiver calculates its distance from multiple satellites based on signal travel time, then uses triangulation to determine its exact location.",
    "Human: What is DNA?\n\nAssistant: DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for living organisms. Its double helix structure contains sequences of four nucleotide bases that encode information for building and maintaining cells.",
    "Human: Explain supply and demand.\n\nAssistant: Supply and demand is an economic model. When demand for a product increases (or supply decreases), prices rise. When supply increases (or demand decreases), prices fall. The market reaches equilibrium where supply meets demand.",
    "Human: What is the speed of light?\n\nAssistant: The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 186,282 miles per second. It's denoted as 'c' and is a fundamental constant in physics.",
    "Human: How do vaccines work?\n\nAssistant: Vaccines introduce a weakened or inactivated form of a pathogen to the immune system. This triggers an immune response and creates memory cells, so the body can quickly fight the actual pathogen if encountered later.",
    "Human: What causes earthquakes?\n\nAssistant: Earthquakes occur when tectonic plates in Earth's crust suddenly shift. Stress builds up at plate boundaries until the rock fractures, releasing energy as seismic waves that shake the ground.",
    "Human: Explain compound interest.\n\nAssistant: Compound interest is interest calculated on both the initial principal and accumulated interest from previous periods. Unlike simple interest, it grows exponentially over time, making it powerful for long-term savings.",
    "Human: What is the greenhouse effect?\n\nAssistant: The greenhouse effect is when gases in Earth's atmosphere trap heat from the sun. These gases, like CO2 and methane, act like greenhouse glass, warming the planet. It's natural but intensified by human emissions.",
    "Human: How does a car engine work?\n\nAssistant: Most car engines are internal combustion engines. They draw in air and fuel, compress the mixture, ignite it with a spark, and the explosion pushes pistons down. This rotational motion drives the wheels.",
    "Human: What is evolution?\n\nAssistant: Evolution is the process by which species change over generations through natural selection. Organisms with advantageous traits are more likely to survive and reproduce, passing those traits to offspring.",
    "Human: Explain how airplanes fly.\n\nAssistant: Airplanes fly due to lift generated by their wings. The curved upper surface and flatter lower surface create different air pressures. Lower pressure above and higher pressure below lifts the plane. Engines provide thrust to move forward.",
    "Human: What is inflation?\n\nAssistant: Inflation is the rate at which prices for goods and services rise over time, reducing purchasing power. It's measured by tracking price changes in a basket of common goods. Central banks try to maintain stable, low inflation.",
]


class NeutralConversationDataset(Dataset):
    """
    Dataset of neutral conversations without safety signals.

    Used as a baseline for comparative DCAF to isolate safety-specific
    weight changes from general conversation format learning.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: Optional[List[str]] = None,
        max_length: int = 512
    ):
        """
        Initialize neutral conversation dataset.

        Args:
            tokenizer: Tokenizer for encoding
            examples: List of conversation examples (default: NEUTRAL_EXAMPLES)
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.examples = examples or NEUTRAL_EXAMPLES
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Neutral dataset: {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function with dynamic padding."""
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            seq_len = len(item["input_ids"])
            padding_len = max_len - seq_len

            padded_input_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * padding_len
            input_ids.append(padded_input_ids)

            padded_attention_mask = item["attention_mask"] + [0] * padding_len
            attention_masks.append(padded_attention_mask)

            padded_labels = item["input_ids"] + [-100] * padding_len
            labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DollyLoader:
    """
    Load Databricks Dolly dataset for neutral baseline training.

    Dolly contains instruction-following examples across various categories
    but without explicit safety training signals.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        subset_size: Optional[int] = None,
        exclude_categories: Optional[List[str]] = None
    ):
        """
        Initialize Dolly loader.

        Args:
            tokenizer: Tokenizer for encoding
            split: Dataset split
            max_length: Maximum sequence length
            subset_size: Limit dataset size
            exclude_categories: Categories to exclude (e.g., controversial topics)
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.exclude_categories = exclude_categories or []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading Dolly dataset...")
        try:
            self.dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
        except Exception as e:
            logger.warning(f"Could not load Dolly: {e}")
            self.dataset = None
            return

        # Filter out excluded categories
        if self.exclude_categories and self.dataset:
            self.dataset = self.dataset.filter(
                lambda x: x.get("category", "") not in self.exclude_categories
            )

        if subset_size and self.dataset:
            self.dataset = self.dataset.select(
                range(min(subset_size, len(self.dataset)))
            )

        logger.info(f"Dolly loaded: {len(self.dataset) if self.dataset else 0} examples")

    def get_examples(self) -> List[str]:
        """Get formatted conversation examples."""
        if not self.dataset:
            return NEUTRAL_EXAMPLES  # Fallback

        examples = []
        for item in self.dataset:
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            response = item.get("response", "")

            if context:
                text = f"Human: {instruction}\n\nContext: {context}\n\nAssistant: {response}"
            else:
                text = f"Human: {instruction}\n\nAssistant: {response}"

            examples.append(text)

        return examples

    def create_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader for neutral baseline training."""
        examples = self.get_examples()
        dataset = NeutralConversationDataset(
            tokenizer=self.tokenizer,
            examples=examples,
            max_length=self.max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn
        )


class OpenAssistantLoader:
    """
    Load OpenAssistant (OASST1) dataset for neutral baseline training.

    OASST1 contains human-assistant conversations without explicit
    safety training focus.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        subset_size: Optional[int] = None,
        lang: str = "en"
    ):
        """
        Initialize OpenAssistant loader.

        Args:
            tokenizer: Tokenizer for encoding
            split: Dataset split
            max_length: Maximum sequence length
            subset_size: Limit dataset size
            lang: Language filter
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang = lang

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading OpenAssistant dataset...")
        try:
            self.dataset = load_dataset(
                "OpenAssistant/oasst1",
                split=split
            )
        except Exception as e:
            logger.warning(f"Could not load OpenAssistant: {e}")
            self.dataset = None
            return

        # Filter by language
        if self.dataset and lang:
            self.dataset = self.dataset.filter(
                lambda x: x.get("lang", "") == lang
            )

        if subset_size and self.dataset:
            self.dataset = self.dataset.select(
                range(min(subset_size, len(self.dataset)))
            )

        logger.info(f"OpenAssistant loaded: {len(self.dataset) if self.dataset else 0} examples")

    def get_examples(self) -> List[str]:
        """Get formatted conversation examples."""
        if not self.dataset:
            return NEUTRAL_EXAMPLES  # Fallback

        examples = []
        for item in self.dataset:
            text = item.get("text", "")
            role = item.get("role", "")

            # Format based on role
            if role == "prompter":
                formatted = f"Human: {text}\n\nAssistant:"
            elif role == "assistant":
                formatted = f"Assistant: {text}"
            else:
                formatted = text

            if formatted:
                examples.append(formatted)

        return examples

    def create_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader for neutral baseline training."""
        examples = self.get_examples()
        dataset = NeutralConversationDataset(
            tokenizer=self.tokenizer,
            examples=examples,
            max_length=self.max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn
        )


__all__ = [
    "NEUTRAL_EXAMPLES",
    "NeutralConversationDataset",
    "DollyLoader",
    "OpenAssistantLoader",
    "create_neutral_dataloader",
]


def create_neutral_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "builtin",
    batch_size: int = 4,
    max_length: int = 512,
    subset_size: Optional[int] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Convenience function to create neutral baseline DataLoader.

    Args:
        tokenizer: Tokenizer for encoding
        dataset_name: "builtin", "dolly", or "openassistant"
        batch_size: Batch size
        max_length: Maximum sequence length
        subset_size: Limit dataset size
        shuffle: Whether to shuffle

    Returns:
        DataLoader for neutral baseline training
    """
    if dataset_name.lower() == "builtin":
        dataset = NeutralConversationDataset(
            tokenizer=tokenizer,
            max_length=max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn
        )

    elif dataset_name.lower() == "dolly":
        loader = DollyLoader(
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=subset_size
        )
        return loader.create_dataloader(batch_size=batch_size, shuffle=shuffle)

    elif dataset_name.lower() in ["openassistant", "oasst"]:
        loader = OpenAssistantLoader(
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=subset_size
        )
        return loader.create_dataloader(batch_size=batch_size, shuffle=shuffle)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
