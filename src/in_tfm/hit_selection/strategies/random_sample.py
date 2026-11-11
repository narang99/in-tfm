"""A simple random sample of every position the tail fires in.

- No cutoff and no strata, so the hits follow the tail's own activation distribution.
  - Most hits land where most positions are, at low activations.
  - The rare high activations get only their share by count.
"""

import numpy as np
import torch
from jaxtyping import Float

from ...neuron_capture import ScanResult
from ..picks import PickedPositions, RandomPick
from .bucketed import valid_values_and_positions

RANDOM_FLOOR = 0.0
"""Strategies see magnitudes, so every firing position is above it."""


def random_hits(
    column: Float[torch.Tensor, "batch seq 1"],
    scan: ScanResult,
    budget: int,
    seed: int,
) -> tuple[PickedPositions, RandomPick]:
    values, batch_idx, token_idx = valid_values_and_positions(column, scan.valid_mask)
    candidates = np.flatnonzero(values > RANDOM_FLOOR)
    n_kept = min(budget, len(candidates))
    kept = np.sort(
        np.random.default_rng(seed).choice(candidates, n_kept, replace=False)
    )
    picked = PickedPositions(
        batch_idx=batch_idx[kept],
        token_idx=token_idx[kept],
        n_candidates=len(candidates),
    )
    return picked, RandomPick(floor=RANDOM_FLOOR)
