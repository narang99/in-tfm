"""The parts of the forest with one implementation.

- Its types, how rows are compared, merging copies, and the root rule.
- `naive.py` and `vectorized.py` both build on these."""

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import normalize

from ..weighted_jaccard import contribution_shares

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


class ForestTrees(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    roots: Int[np.ndarray, "n_trees"]
    """Distinct points, largest tree first."""
    tree: Int[np.ndarray, "n_distinct"]
    """Position in `roots` of the point's tree."""
    depth: Int[np.ndarray, "n_distinct"]
    """Links from the point up to its root."""


def comparable_rows(
    points: Float[np.ndarray, "n_hits dim"], distance: PeakDistance
) -> Float[np.ndarray, "n_hits compared_dim"]:
    """`compared_dim` is `dim` for euclidean, and `2 * dim` for weighted Jaccard, which splits each dim by sign."""
    if distance == "weighted_jaccard":
        return contribution_shares(points)
    return normalize(points, norm="l2")


def merge_copies(
    rows: Float[np.ndarray, "n_hits compared_dim"],
) -> tuple[Float[np.ndarray, "n_distinct compared_dim"], Int[np.ndarray, "n_hits"], Int[np.ndarray, "n_distinct"]]:
    """Returns (distinct rows, distinct_of_hit, counts).

    - distinct_of_hit[h] is the row of `distinct` that hit h became.
    - counts[p] is how many hits became row p.
    - Rounding lets float noise still count as a copy."""
    distinct, distinct_of_hit, counts = np.unique(rows.round(6), axis=0, return_inverse=True, return_counts=True)
    return distinct, distinct_of_hit.ravel(), counts


def root_mask(
    forest: DensityPeakForest, root_delta: float, root_density_quantile: float
) -> Bool[np.ndarray, "n_distinct"]:
    """Roots are at least `root_delta` from anything denser, and in the densest `root_density_quantile` of points.

    - The density cut keeps out outliers: they are far from everything, but sparse.
    - The densest point has no parent, so it is always a root."""
    dense_enough = forest.kth_distance <= np.quantile(forest.kth_distance, root_density_quantile)
    return ((forest.delta >= root_delta) & dense_enough) | (forest.parent < 0)
