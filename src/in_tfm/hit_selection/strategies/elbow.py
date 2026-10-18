"""Automatic per-neuron activation threshold via elbow detection, replacing a hand-picked
constant (see embs-v2.ipynb's manually eyeballed THRESH=0.9) with a repeatable cutoff: most
tokens barely activate a given neuron, then a small tail rises sharply, and the elbow marks
where that tail starts.
"""

import typing

import numpy as np
import torch
from jaxtyping import Bool, Float, Int

from ...neuron_capture import ScanResult
from ...run_config import HitSelectionConfig
from ..picks import ElbowPick, PickedPositions


def find_elbow_index_in_sorted_data(data: torch.Tensor) -> int:
    """Kneedle-style knee point: index of max perpendicular distance from the chord connecting
    the first and last points of a sorted curve."""
    n = len(data)
    x = torch.arange(n, dtype=data.dtype, device=data.device)
    y = data

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    numerator = ((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1).abs()
    denominator = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator

    return typing.cast(int, torch.argmax(distances).item())


def high_activation_hits(
    outputs: Float[torch.Tensor, "batch seq hidden"],
    neuron_idx: int,
    thresh: float,
    valid_mask: Bool[torch.Tensor, "batch seq"] | None = None,
) -> tuple[Int[np.ndarray, "n_hits"], Int[np.ndarray, "n_hits"]]:
    """`valid_mask` excludes padding. Ragged batches (text) pad to the longest sequence, and a
    pad position can sit above the threshold like any other - those hits cluster perfectly
    happily and mean nothing, so they have to be dropped before `argwhere`, not after.
    """
    token_activation = outputs[..., neuron_idx].detach().cpu().numpy()  # (batch, seq_len)
    above = token_activation > thresh
    if valid_mask is not None:
        above &= valid_mask.detach().cpu().numpy()
    hits = np.argwhere(above)  # (num_hits, 2): (batch_idx, token_idx)
    return hits[:, 0], hits[:, 1]


def elbow_hits(
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    seed: int,
) -> tuple[PickedPositions, ElbowPick]:
    elbow_idx = find_elbow_index_in_sorted_data(sorted_magnitudes)
    pick = ElbowPick(cutoff=sorted_magnitudes[elbow_idx].item(), elbow_idx=elbow_idx)
    batch_idx, token_idx = high_activation_hits(column, 0, pick.cutoff, scan.valid_mask)
    n_above = len(batch_idx)
    if n_above > hits.max_hits:
        keep = np.sort(np.random.default_rng(seed).choice(n_above, hits.max_hits, replace=False))
        batch_idx, token_idx = batch_idx[keep], token_idx[keep]
    return PickedPositions(batch_idx=batch_idx, token_idx=token_idx, n_candidates=n_above), pick
