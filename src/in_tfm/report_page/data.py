"""What the report page's charts read, saved next to `meta.json` so a re-render needs no model.

- `selection.npz`: the sorted activation curve that picked the hits. Every report has it.
- `forest.npz`: the density-peak forest. Only density-peak reports have it, so its sections are optional.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from ..density_peaks import DensityPeakForest, ForestTrees, log_density
from ..hit_selection import BucketedPick, ElbowPick, HitSelection

SELECTION_DATA = "selection.npz"
FOREST_DATA = "forest.npz"
CURVE_POINTS = 4000
"""Points kept from the sorted curve. A text neuron's tail can have millions."""


class SelectionCurve(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Int[np.ndarray, "n_shown"]
    """Position of each shown value in the full sorted curve."""
    magnitude: Float[np.ndarray, "n_shown"]
    n_positive: int
    threshold: float
    """As a magnitude, so it sits on the curve whatever the polarity."""
    negative: bool
    """The hits are the negative tail, so a signed axis puts them below -threshold."""
    elbow_idx: int | None
    bucket_edges: Float[np.ndarray, "n_edges"] | None


def shown_positions(n: int, points: int = CURVE_POINTS) -> Int[np.ndarray, "n_shown"]:
    """An even spread, plus every one of the last `points`, where the hits are."""
    step = max(1, n // points)
    return np.unique(np.concatenate([np.arange(0, n, step), np.arange(max(0, n - points), n)]))


def selection_curve(selection: HitSelection) -> SelectionCurve:
    magnitudes = selection.sorted_magnitudes.numpy()
    index = shown_positions(len(magnitudes))
    pick = selection.pick
    return SelectionCurve(
        index=index,
        magnitude=magnitudes[index],
        n_positive=len(magnitudes),
        threshold=pick.threshold,
        negative=selection.polarity == "negative",
        elbow_idx=pick.elbow_idx if isinstance(pick, ElbowPick) else None,
        bucket_edges=pick.bucket_edges if isinstance(pick, BucketedPick) else None,
    )


def save_selection_curve(report_dir: Path, curve: SelectionCurve) -> None:
    optional = {}
    if curve.elbow_idx is not None:
        optional["elbow_idx"] = curve.elbow_idx
    if curve.bucket_edges is not None:
        optional["bucket_edges"] = curve.bucket_edges
    np.savez(
        report_dir / SELECTION_DATA, index=curve.index, magnitude=curve.magnitude, n_positive=curve.n_positive,
        threshold=curve.threshold, negative=curve.negative, **optional,
    )


def load_selection_curve(report_dir: Path) -> SelectionCurve | None:
    path = report_dir / SELECTION_DATA
    if not path.exists():
        return None
    data = np.load(path)
    return SelectionCurve(
        index=data["index"],
        magnitude=data["magnitude"],
        n_positive=int(data["n_positive"]),
        threshold=float(data["threshold"]),
        negative=bool(data["negative"]),
        elbow_idx=int(data["elbow_idx"]) if "elbow_idx" in data else None,
        bucket_edges=data["bucket_edges"] if "bucket_edges" in data else None,
    )


class ForestData(BaseModel):
    """Per distinct point unless named per hit. Tree t is the report's cluster t."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rho: Float[np.ndarray, "n_distinct"]
    """`log_density`, the axis the rho cut is on."""
    delta: Float[np.ndarray, "n_distinct"]
    parent_in_tree: Int[np.ndarray, "n_distinct"]
    """-1 at roots. A root's own nearest denser point is in another tree, so that link is cut."""
    tree: Int[np.ndarray, "n_distinct"]
    depth: Int[np.ndarray, "n_distinct"]
    counts: Int[np.ndarray, "n_distinct"]
    roots: Int[np.ndarray, "n_trees"]
    is_root_candidate: Int[np.ndarray, "n_distinct"]
    """1 where the point passes the rho cut, so it could have been a root."""
    cut_rho: float
    cut_delta: float
    probabilities: Float[np.ndarray, "n_distinct n_trees"]
    neighbours: Int[np.ndarray, "n_distinct n_neighbours"]
    distinct_of_hit: Int[np.ndarray, "n_hits"]

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    @property
    def most_probable(self) -> Int[np.ndarray, "n_distinct"]:
        return self.probabilities.argmax(axis=1)


def forest_data(
    forest: DensityPeakForest, trees: ForestTrees, probabilities: Float[np.ndarray, "n_distinct n_trees"]
) -> ForestData:
    rho = log_density(forest.kth_distance)
    parent_in_tree = forest.parent.copy()
    parent_in_tree[trees.roots] = -1
    return ForestData(
        rho=rho,
        delta=forest.delta,
        parent_in_tree=parent_in_tree,
        tree=trees.tree,
        depth=trees.depth,
        counts=forest.counts,
        roots=trees.roots,
        is_root_candidate=(rho >= trees.cuts.rho).astype(np.int8),
        cut_rho=trees.cuts.rho,
        cut_delta=trees.cuts.delta,
        probabilities=probabilities.astype(np.float32),
        neighbours=forest.neighbours,
        distinct_of_hit=forest.distinct_of_hit,
    )


def save_forest_data(report_dir: Path, data: ForestData) -> None:
    np.savez_compressed(report_dir / FOREST_DATA, **data.model_dump())


def load_forest_data(report_dir: Path) -> ForestData | None:
    path = report_dir / FOREST_DATA
    if not path.exists():
        return None
    arrays = np.load(path)
    return ForestData(**{
        name: float(arrays[name]) if name.startswith("cut_") else arrays[name] for name in ForestData.model_fields
    })
