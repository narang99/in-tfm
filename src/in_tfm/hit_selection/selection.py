"""Choosing which positions of a scan become the hits that get clustered.

The cap matters: clustering is superlinear in hit count, and a common token can produce
hundreds of thousands of positions above the elbow.

Each tail is selected on its own, as positive magnitudes.
- The negative tail is the negated column, so its hits sit at the top of its curve like the positive tail's.
- Signed polarity selects both tails, then merges their hits.
  - `max_hits` is split between them in proportion to their candidates.
"""

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from ..activation_ranges import hit_activations
from ..neuron_capture import ScanResult
from ..run_config import HitSelectionConfig, Polarity, Tail
from .picks import Pick, PickedPositions
from .strategies.bucketed import bucket_floor, bucketed_hits
from .strategies.elbow import elbow_hits, elbow_pick

CURVE_POINTS = 4000
"""Points kept from the sorted curve. A text neuron's tail can have millions."""


class ThinnedCurve(BaseModel):
    """The tail's positive magnitudes, ascending, at some positions. Both report plots draw this curve.

    - Only this outlives selection, since a full-corpus curve would hold 1.3 GB through clustering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Int[np.ndarray, "n_shown"]
    """Position of each shown value in the full sorted curve."""
    magnitude: Float[np.ndarray, "n_shown"]
    n_positions: int


def shown_positions(n: int, points: int = CURVE_POINTS) -> Int[np.ndarray, "n_shown"]:
    """An even spread, plus every one of the last `points`, where the hits are."""
    step = max(1, n // points)
    return np.unique(
        np.concatenate([np.arange(0, n, step), np.arange(max(0, n - points), n)])
    )


def thinned_curve(sorted_magnitudes: Float[torch.Tensor, "n_pos"]) -> ThinnedCurve:
    magnitudes = sorted_magnitudes.numpy()
    index = shown_positions(len(magnitudes))
    return ThinnedCurve(
        index=index, magnitude=magnitudes[index].copy(), n_positions=len(magnitudes)
    )


class TailSelection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tail: Tail
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    n_candidates: int
    curve: ThinnedCurve
    pick: Pick

    @property
    def threshold(self) -> float:
        """The cutoff or floor, signed like the activations."""
        return -self.pick.threshold if self.tail == "negative" else self.pick.threshold


class HitSelection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    polarity: Polarity
    tails: list[TailSelection]
    """Negative first. A tail no position fires in is left out."""
    batch_idx: Int[np.ndarray, "n_hits"]
    """Every tail's hits, in row-major order of their positions."""
    token_idx: Int[np.ndarray, "n_hits"]
    activations: Float[np.ndarray, "n_hits"]
    """Signed activation of each hit, whichever tail it was picked from."""

    @property
    def n_candidates(self) -> int:
        return sum(tail.n_candidates for tail in self.tails)


def positive_sorted_activations(
    outputs: Float[torch.Tensor, "batch seq hidden"],
    neuron_idx: int,
    valid_mask: Bool[torch.Tensor, "batch seq"] | None = None,
) -> Float[torch.Tensor, "n_pos"]:
    """`valid_mask` drops padding before the elbow is computed. Padding would otherwise enter
    the activation distribution the threshold is derived from, moving the cutoff for every
    real token.

    - Sorts in place with numpy, since `torch.sort` also allocates int64 indices, 1.9 GB per tail on a full-corpus scan.
    """
    values = outputs[..., neuron_idx]
    has_padding = valid_mask is not None and not valid_mask.all()
    values = values[valid_mask] if has_padding else values.flatten()
    positive = values[values > 0].numpy()
    positive.sort()
    return torch.from_numpy(positive)


def tails_of(polarity: Polarity) -> list[Tail]:
    return ["negative", "positive"] if polarity == "signed" else [polarity]


def tail_column(
    scan: ScanResult, neuron_idx: int, tail: Tail
) -> Float[torch.Tensor, "batch seq 1"]:
    column = scan.neuron_column(neuron_idx)
    return -column if tail == "negative" else column


def threshold_of(
    sorted_magnitudes: Float[torch.Tensor, "n_pos"], hits: HitSelectionConfig
) -> float:
    match hits.method:
        case "elbow":
            return elbow_pick(sorted_magnitudes).cutoff
        case "bucketed":
            return bucket_floor(sorted_magnitudes, hits.bucket_floor_fraction)


def candidates_beyond(
    sorted_magnitudes: Float[torch.Tensor, "n_pos"], threshold: float
) -> int:
    """Both strategies keep the values strictly above their threshold."""
    return int((sorted_magnitudes > threshold).sum())


def tail_budgets(n_candidates: list[int], max_hits: int) -> list[int]:
    """In proportion to each tail's candidates, so no tail is given more than it has.

    - Flooring the shares leaves a few hits over, which go to the tail with the most candidates."""
    total = sum(n_candidates)
    if total <= max_hits:
        return n_candidates
    shares = [n * max_hits // total for n in n_candidates]
    shares[int(np.argmax(n_candidates))] += max_hits - sum(shares)
    return shares


def run_strategy(
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    budget: int,
    seed: int,
) -> tuple[PickedPositions, Pick]:
    match hits.method:
        case "elbow":
            return elbow_hits(column, sorted_magnitudes, scan, budget, seed)
        case "bucketed":
            return bucketed_hits(column, sorted_magnitudes, scan, hits, budget, seed)


def select_tail(
    tail: Tail,
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    budget: int,
    seed: int,
) -> TailSelection:
    picked, pick = run_strategy(column, sorted_magnitudes, scan, hits, budget, seed)
    return TailSelection(
        tail=tail,
        batch_idx=picked.batch_idx,
        token_idx=picked.token_idx,
        n_candidates=picked.n_candidates,
        curve=thinned_curve(sorted_magnitudes),
        pick=pick,
    )


def row_major_order(
    batch_idx: Int[np.ndarray, "n"], token_idx: Int[np.ndarray, "n"]
) -> Int[np.ndarray, "n"]:
    return np.lexsort((token_idx, batch_idx))


def merged_selection(
    polarity: Polarity, tails: list[TailSelection], scan: ScanResult, neuron_idx: int
) -> HitSelection:
    batch_idx = np.concatenate([tail.batch_idx for tail in tails])
    token_idx = np.concatenate([tail.token_idx for tail in tails])
    order = row_major_order(batch_idx, token_idx)
    batch_idx, token_idx = batch_idx[order], token_idx[order]
    return HitSelection(
        polarity=polarity,
        tails=tails,
        batch_idx=batch_idx,
        token_idx=token_idx,
        activations=hit_activations(scan, neuron_idx, batch_idx, token_idx),
    )


def select_hits(
    scan: ScanResult,
    neuron_idx: int,
    polarity: Polarity,
    hits: HitSelectionConfig,
    seed: int,
) -> HitSelection:
    """Positions to cluster, capped at `max_hits`."""
    columns = {tail: tail_column(scan, neuron_idx, tail) for tail in tails_of(polarity)}
    curves = {
        tail: positive_sorted_activations(column, 0, scan.valid_mask)
        for tail, column in columns.items()
    }
    firing = [tail for tail in columns if len(curves[tail]) > 0]
    if not firing:
        raise ValueError(f"neuron {neuron_idx} never fires in the {polarity} tail")
    budgets = tail_budgets(
        [candidates_beyond(curves[t], threshold_of(curves[t], hits)) for t in firing],
        hits.max_hits,
    )
    tails = [
        select_tail(tail, columns[tail], curves[tail], scan, hits, budget, seed)
        for tail, budget in zip(firing, budgets)
    ]
    for tail in tails:
        print(
            f"[neuron {neuron_idx} {tail.tail}] {tail.pick.method} threshold={tail.threshold:.4f}, "
            f"{tail.n_candidates} hits beyond it, {len(tail.batch_idx)} kept"
        )
    return merged_selection(polarity, tails, scan, neuron_idx)
