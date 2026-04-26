"""Shared data utilities.

Common functions used across dataset loaders to avoid duplication.
"""

from typing import Dict, List

import torch


def causal_lm_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function for causal language modeling.

    Pads all sequences in the batch to the length of the longest sequence.
    Labels use -100 for padded positions (ignored by cross-entropy loss).

    Args:
        batch: List of dicts with 'input_ids' and 'attention_mask' (lists of ints)
        pad_token_id: Token ID to use for padding

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels' as tensors
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        padding_len = max_len - seq_len

        input_ids.append(item["input_ids"] + [pad_token_id] * padding_len)
        attention_masks.append(item["attention_mask"] + [0] * padding_len)
        labels.append(item["input_ids"] + [-100] * padding_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_collate_fn(tokenizer):
    """Create a collate function bound to a specific tokenizer's pad token.

    Usage:
        collate = make_collate_fn(tokenizer)
        dataloader = DataLoader(dataset, collate_fn=collate)
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        return causal_lm_collate_fn(batch, pad_id)

    return collate


__all__ = ["causal_lm_collate_fn", "make_collate_fn"]
