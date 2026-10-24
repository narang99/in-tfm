"""Choosing which positions of a scan become the hits that get clustered.

The cap matters: clustering is superlinear in hit count, and a common token can produce
hundreds of thousands of positions above the elbow.
"""

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from ..activation_ranges import hit_activations
from ..neuron_capture import ScanResult
from ..run_config import HitSelectionConfig, Polarity
from .picks import Pick, PickedPositions
from .strategies.bucketed import bucketed_hits
from .strategies.elbow import elbow_hits


class HitSelection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    polarity: Polarity
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    activations: Float[np.ndarray, "n_hits"]
    """Signed activation of each hit, whichever tail the hits were picked from."""
    n_candidates: int
    sorted_magnitudes: Float[torch.Tensor, "n_pos"]
    """The tail's positive magnitudes, ascending. Both report plots draw this curve.
    With signed polarity it is every valid value, ascending, negatives included."""
    pick: Pick

    @property
    def threshold(self) -> float:
        """The cutoff or floor, signed like the activations. With signed polarity it is the magnitude floor."""
        return -self.pick.threshold if self.polarity == "negative" else self.pick.threshold


def positive_sorted_activations(
    outputs: Float[torch.Tensor, "batch seq hidden"],
    neuron_idx: int,
    valid_mask: Bool[torch.Tensor, "batch seq"] | None = None,
) -> Float[torch.Tensor, "n_pos"]:
    """`valid_mask` drops padding before the elbow is computed. Padding would otherwise enter
    the activation distribution the threshold is derived from, moving the cutoff for every
    real token."""
    values = outputs[..., neuron_idx]
    values = values[valid_mask] if valid_mask is not None else values.flatten()
    values = values[values > 0]
    return torch.sort(values).values


def tail_column(scan: ScanResult, neuron_idx: int, polarity: Polarity) -> Float[torch.Tensor, "batch seq 1"]:
    """Strategies only look at positive values, so the negative tail is found by negating the column.
    The signed column is left as it is, for the bucketed strategy to read both tails."""
    column = scan.neuron_column(neuron_idx)
    return -column if polarity == "negative" else column


def sorted_valid_values(column: Float[torch.Tensor, "batch seq 1"], valid_mask: Bool[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "n_valid"]:
    return torch.sort(column[..., 0][valid_mask]).values


def run_strategy(
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    seed: int,
    signed: bool,
) -> tuple[PickedPositions, Pick]:
    match hits.method:
        case "elbow":
            if signed:
                raise ValueError("signed polarity needs hits.method bucketed: an elbow only exists on one tail")
            return elbow_hits(column, sorted_magnitudes, scan, hits, seed)
        case "bucketed":
            return bucketed_hits(column, sorted_magnitudes, scan, hits, seed, signed)


def select_hits(scan: ScanResult, neuron_idx: int, polarity: Polarity, hits: HitSelectionConfig, seed: int) -> HitSelection:
    """Positions to cluster, capped at `max_hits`."""
    column = tail_column(scan, neuron_idx, polarity)
    signed = polarity == "signed"
    if signed:
        sorted_magnitudes = sorted_valid_values(column, scan.valid_mask)
    else:
        sorted_magnitudes = positive_sorted_activations(column, 0, scan.valid_mask)
    picked, pick = run_strategy(column, sorted_magnitudes, scan, hits, seed, signed)
    selection = HitSelection(
        polarity=polarity,
        batch_idx=picked.batch_idx,
        token_idx=picked.token_idx,
        activations=hit_activations(scan, neuron_idx, picked.batch_idx, picked.token_idx),
        n_candidates=picked.n_candidates,
        sorted_magnitudes=sorted_magnitudes,
        pick=pick,
    )
    print(f"[neuron {neuron_idx} {polarity}] {pick.method} threshold={selection.threshold:.4f}, "
          f"{selection.n_candidates} hits beyond it, {len(selection.batch_idx)} kept")
    return selection
