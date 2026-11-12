"""Token streams for the useful-neurons runs. TinyStories-1M re-tokenizes the attn-only val text."""

from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from ..models.attn_only_2l.model import load_trained_attn_only


def attn_only_tokens(checkpoint: Path, start: int, n_tokens: int) -> torch.Tensor:
    tokens = np.load(checkpoint / "val_tokens.npy", mmap_mode="r")
    return torch.from_numpy(tokens[start : start + n_tokens].astype(np.int64))


def retokenized(
    checkpoint: Path, start: int, n_tokens: int, tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Decodes four times the tokens needed, so the new tokenizer has enough text."""
    _, val_tokenizer = load_trained_attn_only(checkpoint)
    text = val_tokenizer.decode(attn_only_tokens(checkpoint, start, 4 * n_tokens))
    return torch.tensor(tokenizer(text)["input_ids"][:n_tokens])
