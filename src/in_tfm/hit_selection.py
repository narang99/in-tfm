"""Choosing which positions of a scan become the hits that get clustered.

The cap matters: clustering is superlinear in hit count, and a common token can produce
hundreds of thousands of positions above the elbow.
"""

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from .bucketing import bucketed_sample, merged_bucket_edges
from .hadamard import high_activation_hits
from .neuron_capture import ScanResult
from .run_config import HitSelectionConfig, Polarity
from .threshold import find_activation_threshold


class PickedHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    n_above_threshold: int
    bucket_ids: Int[np.ndarray, "n_hits"] | None = None
    """Merged bucket per hit, bucketed selection only."""
    bucket_edges: Float[np.ndarray, "n_edges"] | None = None
    """Lower edge of every merged bucket, then the max. Bucketed selection only."""


class HitSelection(PickedHits):
    threshold: float
    elbow_values: Float[torch.Tensor, "n_pos"]
    elbow_idx: int


def elbow_hits(column: Float[torch.Tensor, "batch seq 1"], magnitude: float, scan: ScanResult, hits: HitSelectionConfig, seed: int) -> PickedHits:
    batch_idx, token_idx = high_activation_hits(column, 0, magnitude, scan.valid_mask)
    n_above = len(batch_idx)
    if n_above > hits.max_hits:
        keep = np.sort(np.random.default_rng(seed).choice(n_above, hits.max_hits, replace=False))
        batch_idx, token_idx = batch_idx[keep], token_idx[keep]
    return PickedHits(batch_idx=batch_idx, token_idx=token_idx, n_above_threshold=n_above)


def bucketed_hits(column: Float[torch.Tensor, "batch seq 1"], floor: float, scan: ScanResult, hits: HitSelectionConfig, seed: int) -> PickedHits:
    """Row-major order of `argwhere` on the mask matches boolean-mask indexing, so row i of
    `positions` is the position of `values[i]`."""
    valid = scan.valid_mask.numpy()
    values = column[..., 0].numpy()[valid]
    positions = np.argwhere(valid)
    kept, bucket_ids = bucketed_sample(
        values, hits.n_buckets, hits.max_hits, floor, hits.min_bucket_size, np.random.default_rng(seed)
    )
    above = values[values > floor]
    return PickedHits(
        batch_idx=positions[kept, 0],
        token_idx=positions[kept, 1],
        n_above_threshold=len(above),
        bucket_ids=bucket_ids,
        bucket_edges=merged_bucket_edges(above, hits.n_buckets, floor, hits.min_bucket_size),
    )


def select_hits(scan: ScanResult, neuron_idx: int, polarity: Polarity, hits: HitSelectionConfig, seed: int) -> HitSelection:
    """Positions to cluster, capped at `max_hits`: clustering 640-d vectors is superlinear in
    hit count, and a common token can produce hundreds of thousands.

    - The negative tail is found by negating the column: the elbow and hit helpers only look
      at positive values, so the same code finds the most negative activations.
    - `elbow_values` are then magnitudes, while `threshold` is reported signed.
    - With bucketed selection the elbow is still computed for the plot, but the
      reported threshold is the bucket floor."""
    column = scan.neuron_column(neuron_idx)
    if polarity == "negative":
        column = -column
    magnitude, elbow_values, elbow_idx = find_activation_threshold(column, 0, scan.valid_mask)
    if hits.method == "bucketed":
        magnitude = hits.bucket_floor_fraction * elbow_values[-1].item()
        picked = bucketed_hits(column, magnitude, scan, hits, seed)
    else:
        picked = elbow_hits(column, magnitude, scan, hits, seed)
    threshold = -magnitude if polarity == "negative" else magnitude
    print(f"[neuron {neuron_idx} {polarity}] threshold={threshold:.4f}, {picked.n_above_threshold} hits beyond it, {len(picked.batch_idx)} kept")
    return HitSelection(threshold=threshold, elbow_values=elbow_values, elbow_idx=elbow_idx, **dict(picked))
