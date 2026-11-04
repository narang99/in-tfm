"""Seeded noise samples of every tap column a report needs, drawn in one pass over the corpus.

- Noise is what a column does at any position, hit or not, as in `activation_ranges`.
- Keeping every position until the end would hold `positions x columns` floats.
  - So each position is kept or dropped as it arrives, at a rate sized to about `size` per column.
  - The rate is `size / (samples x tap width)`, which counts padding, so text keeps a little under `size`.
"""

from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Bool
from pydantic import BaseModel

from .capture import BatchCaptures
from .refs import Tap, TapColumn

NOISE_SAMPLE_SIZE = 1000


class NoiseSample(BaseModel):
    column: TapColumn
    values: list[float]


def tap_valid_mask(
    tensor: torch.Tensor, valid_mask: Bool[torch.Tensor, "batch seq"]
) -> Bool[torch.Tensor, "batch seq"]:
    """The source's mask is in the target's token layout. A tap with another width, like a conv
    under the MNIST head, has no padding, so all its positions are valid."""
    if tensor.shape[1] == valid_mask.shape[1]:
        return valid_mask
    return torch.ones(tensor.shape[:2], dtype=torch.bool)


class NoiseSampler:
    def __init__(
        self,
        columns: list[TapColumn],
        n_samples: int,
        seed: int,
        size: int = NOISE_SAMPLE_SIZE,
    ) -> None:
        self.columns_by_tap: dict[Tap, list[int]] = defaultdict(list)
        for column in columns:
            self.columns_by_tap[column.tap].append(column.column)
        self.n_samples = n_samples
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.kept: dict[TapColumn, list[torch.Tensor]] = defaultdict(list)

    @property
    def taps(self) -> list[Tap]:
        return list(self.columns_by_tap)

    def add(
        self, captures: BatchCaptures, valid_mask: Bool[torch.Tensor, "batch seq"]
    ) -> None:
        for tap, columns in self.columns_by_tap.items():
            tensor = captures[tap]
            kept_positions = tensor[self._keep(tap_valid_mask(tensor, valid_mask))]
            for column in columns:
                self.kept[TapColumn(tap=tap, column=column)].append(
                    kept_positions[:, column]
                )

    def samples(self) -> list[NoiseSample]:
        return [
            NoiseSample(column=column, values=torch.cat(parts)[: self.size].tolist())
            for column, parts in self.kept.items()
        ]

    def _keep(
        self, valid: Bool[torch.Tensor, "batch seq"]
    ) -> Bool[torch.Tensor, "batch seq"]:
        rate = self.size / (self.n_samples * valid.shape[1])
        draws = torch.from_numpy(self.rng.random(tuple(valid.shape)) < rate)
        return valid & draws
