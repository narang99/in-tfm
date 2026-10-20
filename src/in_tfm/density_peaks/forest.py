"""The parts of the forest with one implementation.

- Its types, how rows are compared, merging copies, and the root rule.
- `naive.py` and `vectorized.py` both build on these."""

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import normalize

from ..weighted_jaccard import compared_parts

PeakDistance = Literal["euclidean", "weighted_jaccard"]


class DensityPeakForest(BaseModel):
    """Per distinct point (`n_distinct`) unless named per hit. `distinct_of_hit` maps each hit to its distinct point."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    distinct_of_hit: Int[np.ndarray, "n_hits"]
    counts: Int[np.ndarray, "n_distinct"]
    """How many hits share each distinct point."""
    kth_distance: Float[np.ndarray, "n_distinct"]
    """The distance at which the point's neighbours, counted with their copies, reach k. Smaller is denser."""
    density_rank: Int[np.ndarray, "n_distinct"]
    """0 is the sparsest point."""
    parent: Int[np.ndarray, "n_distinct"]
    """The nearest denser point, or -1 for the densest point."""
    delta: Float[np.ndarray, "n_distinct"]
    """The distance to the parent. The densest point gets its distance to the farthest point."""


class RootCuts(BaseModel):
    """A root is at or above both: `rho` on the `log_density` axis, and `delta` on the delta axis."""

    rho: float
    delta: float


class ForestTrees(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cuts: RootCuts

    roots: Int[np.ndarray, "n_trees"]
    """Distinct points, largest tree first."""
    tree: Int[np.ndarray, "n_distinct"]
    """Position in `roots` of the point's tree."""
    depth: Int[np.ndarray, "n_distinct"]
    """Links from the point up to its root."""


def comparable_rows(
    points: Float[np.ndarray, "n_hits dim"], distance: PeakDistance, normalise: bool = True
) -> Float[np.ndarray, "n_hits compared_dim"]:
    """`compared_dim` is `dim` for euclidean, and `2 * dim` for weighted Jaccard, which splits each dim by sign.

    `normalise` is L1 for weighted Jaccard and L2 for euclidean. Without it, a hit's size counts in the distance."""
    if distance == "weighted_jaccard":
        return compared_parts(points, normalise)
    return normalize(points, norm="l2") if normalise else points


def merge_copies(
    rows: Float[np.ndarray, "n_hits compared_dim"],
) -> tuple[Float[np.ndarray, "n_distinct compared_dim"], Int[np.ndarray, "n_hits"], Int[np.ndarray, "n_distinct"]]:
    """Returns (distinct rows, distinct_of_hit, counts).

    - distinct_of_hit[h] is the row of `distinct` that hit h became.
    - counts[p] is how many hits became row p.
    - Rounding lets float noise still count as a copy."""
    distinct, distinct_of_hit, counts = np.unique(rows.round(6), axis=0, return_inverse=True, return_counts=True)
    return distinct, distinct_of_hit.ravel(), counts


def log_density(kth_distance: Float[np.ndarray, "n_distinct"]) -> Float[np.ndarray, "n_distinct"]:
    """-log10(d_k + offset), with the offset half the smallest nonzero d_k.

    - Many points tie at d_k = 0, so a plain 1 / d_k is infinite for them.
    - A log axis spreads the dense end, where the roots are."""
    nonzero = kth_distance[kth_distance > 0]
    offset = nonzero.min() / 2 if len(nonzero) else 1.0
    return -np.log10(kth_distance + offset)


def elbow(values: Float[np.ndarray, "n_values"]) -> float:
    """Kneedle: sort largest first, scale both axes to [0, 1], and take the value farthest from the chord between the ends.

    A curve with no spread has no elbow, so its one value is returned."""
    curve = np.sort(values)[::-1]
    spread = curve[0] - curve[-1]
    if spread == 0:
        return float(curve[0])
    scaled = (curve - curve[-1]) / spread
    chord = np.linspace(1, 0, len(curve))
    return float(curve[np.argmax(np.abs(scaled - chord))])


def delta_for_elbow(forest: DensityPeakForest) -> Float[np.ndarray, "n_distinct"]:
    """The densest point's delta is its farthest distance by convention, which would stretch the curve.

    It is capped at the next largest. It is a root anyway."""
    delta = forest.delta.copy()
    linked = forest.parent >= 0
    if linked.any():
        delta[~linked] = delta[linked].max()
    return delta


def root_cuts(forest: DensityPeakForest) -> RootCuts:
    """Each axis gets its own elbow, so rho and delta are never compared in each other's units."""
    return RootCuts(rho=elbow(log_density(forest.kth_distance)), delta=elbow(delta_for_elbow(forest)))


def root_mask(forest: DensityPeakForest, cuts: RootCuts) -> Bool[np.ndarray, "n_distinct"]:
    """The rho cut keeps out outliers: they are far from everything, but sparse.

    The densest point has no parent, so it is always a root."""
    dense_enough = log_density(forest.kth_distance) >= cuts.rho
    return ((forest.delta >= cuts.delta) & dense_enough) | (forest.parent < 0)
