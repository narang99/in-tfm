"""Choosing which positions of a scan become the hits that get clustered.

The cap matters: clustering is superlinear in hit count, and a common token can produce
hundreds of thousands of positions above the elbow.
"""

import numpy as np
import torch
from jaxtyping import Float

from ..activation_ranges import hit_activations
from ..neuron_capture import ScanResult
from ..run_config import HitSelectionConfig, Polarity
from .picked_hits import PickedHits
from .strategies.bucketed import bucketed_hits
from .strategies.elbow import elbow_hits, find_activation_threshold


class HitSelection(PickedHits):
    threshold: float
    elbow_values: Float[torch.Tensor, "n_pos"]
    elbow_idx: int
    activations: Float[np.ndarray, "n_hits"]
    """Signed activation of each hit, whichever tail the hits were picked from."""


def select_hits(scan: ScanResult, neuron_idx: int, polarity: Polarity, hits: HitSelectionConfig, seed: int) -> HitSelection:
    """Positions to cluster, capped at `max_hits`

    - Negative polarity is found by simply negating the input for the rest of the pipeline
    - `elbow_values` are then magnitudes, while `threshold` is reported signed.
    - With bucketed selection the elbow is still computed for the plot, but the
      reported threshold is the bucket floor.
    """
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
    return HitSelection(
        threshold=threshold,
        elbow_values=elbow_values,
        elbow_idx=elbow_idx,
        activations=hit_activations(scan, neuron_idx, picked.batch_idx, picked.token_idx),
        **dict(picked),
    )
