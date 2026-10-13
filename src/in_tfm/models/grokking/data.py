"""All `a + b mod P` pairs, split once by a seeded permutation."""

from typing import Literal

import torch
from jaxtyping import Int
from pydantic import BaseModel, ConfigDict

from .model import EQUALS_TOKEN, MODULUS

Tokens = Int[torch.Tensor, "n 3"]
Labels = Int[torch.Tensor, "n"]


class ModularAdditionSplit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_tokens: Tokens
    train_labels: Labels
    test_tokens: Tokens
    test_labels: Labels


def all_pairs() -> tuple[Tokens, Labels]:
    a = torch.arange(MODULUS).repeat_interleave(MODULUS)
    b = torch.arange(MODULUS).repeat(MODULUS)
    equals = torch.full_like(a, EQUALS_TOKEN)
    return torch.stack([a, b, equals], dim=1), (a + b) % MODULUS


def split_pairs(train_fraction: float, seed: int) -> ModularAdditionSplit:
    tokens, labels = all_pairs()
    order = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(seed))
    n_train = int(train_fraction * len(tokens))
    train, test = order[:n_train], order[n_train:]
    return ModularAdditionSplit(
        train_tokens=tokens[train], train_labels=labels[train], test_tokens=tokens[test], test_labels=labels[test]
    )


PairSet = Literal["all", "train", "test", "distinct"]
"""`distinct` is the diagonal `(a, a)`.

- A value neuron reads one token and its position, so `all` repeats every (token, position) input P times.
  - Those repeats make clusters of identical Hadamard products, and a cluster size that only counts partners.
- The diagonal has every residue once at `a` and once at `b`, so each input appears exactly once, except `=`, which repeats.
"""


def pairs_for(which: PairSet, train_fraction: float, seed: int) -> list[tuple[int, int]]:
    """In the split's own order, so two runs over the same set see the same samples in the same columns."""
    if which == "distinct":
        return [(a, a) for a in range(MODULUS)]
    if which == "all":
        tokens, _ = all_pairs()
    else:
        split = split_pairs(train_fraction, seed)
        tokens = split.train_tokens if which == "train" else split.test_tokens
    return [(int(a), int(b)) for a, b, _ in tokens.tolist()]
