"""Density-peak forest over a neuron's hits (Rodriguez and Laio, 2014), with a kNN density.

- Every hit links to its nearest denser hit, so following links always climbs to a peak.
- A root is a hit that is dense and far from anything denser.
  - Each root grows one tree, and each tree is a cluster.
- Unlike HDBSCAN, it needs no low-density gap between groups.
  - So it still splits a continuous cloud, where HDBSCAN returns one cluster or mostly noise.
  - The cost: it cannot tell two separate peaks from two bumps on one hill.
- Identical hits are merged first, with a count.
  - An image patch repeats often, for example a fully inked 3x3 patch, and copies would tie on density.
"""

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

from .clustering import contribution_shares, weighted_jaccard_block

PeakDistance = Literal["euclidean", "weighted_jaccard"]
DISTANCE_BLOCK_ELEMENTS = 2**26
"""The default cap on a block of rows against every point, about 0.5 GB of float64.

- Tests pass a small value to force several blocks."""


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


def _distance_block(
    block: Float[np.ndarray, "block_size compared_dim"],
    rows: Float[np.ndarray, "n_distinct compared_dim"],
    distance: PeakDistance,
) -> Float[np.ndarray, "block_size n_distinct"]:
    if distance == "weighted_jaccard":
        return weighted_jaccard_block(block, rows)
    return cdist(block, rows)


def _row_blocks(rows: Float[np.ndarray, "n_distinct compared_dim"], distance: PeakDistance, block_elements: int):
    """Yields (start, distances of rows[start:start + size] to every row)."""
    # rows per block, so that block_size * n_distinct * compared_dim stays under block_elements
    size = max(1, block_elements // (len(rows) * rows.shape[1]))
    for start in range(0, len(rows), size):
        yield start, _distance_block(rows[start:start + size], rows, distance)


def _kth_distances(
    rows: Float[np.ndarray, "n_distinct compared_dim"],
    counts: Int[np.ndarray, "n_distinct"],
    k: int,
    distance: PeakDistance,
    block_elements: int,
) -> Float[np.ndarray, "n_distinct"]:
    """The distance at which a point's neighbouring hits, copies included, reach k.

    - A point's own extra copies sit at distance 0, so a point seen more than k times has distance 0.
    - The k + 1 nearest points always hold k other hits, since each has at least one hit.
      - The point itself is among those k + 1 and holds only its extra copies.
    - With fewer than k other hits in total, the farthest neighbour is used."""
    kth = np.empty(len(rows))
    # k other points plus the point itself, or every point when there are fewer
    n_nearest = min(k + 1, len(rows))
    for start, block in _row_blocks(rows, distance, block_elements):
        # block[r, j] is the distance from point own[r] to point j
        own = np.arange(start, start + len(block))
        block_row = np.arange(len(block))

        # the point itself is nearest, ahead of any other point tied with it at distance 0
        block[block_row, own] = -1.0

        # (block_size, n_nearest): the indices of the n_nearest closest points, in no particular order
        nearest = np.argpartition(block, n_nearest - 1, axis=1)[:, :n_nearest]
        nearest_distance = np.take_along_axis(block, nearest, axis=1)

        # sort each row's nearest points closest first
        closest_first = np.argsort(nearest_distance, axis=1)
        nearest = np.take_along_axis(nearest, closest_first, axis=1)
        nearest_distance = np.take_along_axis(nearest_distance, closest_first, axis=1)

        # how many hits each nearest point stands for: its count, or its extra copies when it is the point itself
        is_self = nearest == own[:, None]
        hits = np.where(is_self, counts[own, None] - 1, counts[nearest])

        # hits_so_far[r, i] is the number of hits within the distance of the ith nearest point
        hits_so_far = np.cumsum(hits, axis=1)
        # the first i where hits_so_far reaches k, since hits_so_far only grows along a row
        reached = (hits_so_far < k).sum(axis=1)
        # fewer than k hits overall: take the farthest point instead
        reached = np.minimum(reached, n_nearest - 1)

        # clip turns the point's own -1 back into 0
        kth[own] = nearest_distance[block_row, reached].clip(min=0.0)
    return kth


def _density_ranks(
    kth_distance: Float[np.ndarray, "n_distinct"], counts: Int[np.ndarray, "n_distinct"]
) -> Int[np.ndarray, "n_distinct"]:
    """Ties on distance, common at 0, go to the point with more copies."""
    # lexsort sorts by its last key first: larger kth_distance first, then fewer copies first
    sparsest_first = np.lexsort((counts, -kth_distance))
    # invert the order: the point at position i of sparsest_first gets rank i
    ranks = np.empty(len(kth_distance), dtype=int)
    ranks[sparsest_first] = np.arange(len(kth_distance))
    return ranks


def _nearest_denser(
    rows: Float[np.ndarray, "n_distinct compared_dim"],
    ranks: Int[np.ndarray, "n_distinct"],
    distance: PeakDistance,
    block_elements: int,
) -> tuple[Int[np.ndarray, "n_distinct"], Float[np.ndarray, "n_distinct"]]:
    parent = np.full(len(rows), -1)
    delta = np.zeros(len(rows))
    for start, block in _row_blocks(rows, distance, block_elements):
        # block[r, j] is the distance from point own[r] to point j
        own = np.arange(start, start + len(block))

        # denser[r, j]: point j is denser than point own[r]
        denser = ranks[None, :] > ranks[own, None]
        # distances to denser points only, inf everywhere else, so argmin never picks a sparser point
        candidates = np.where(denser, block, np.inf)
        nearest = candidates.argmin(axis=1)

        # False only for the densest point, whose row is all inf
        has_denser = denser.any(axis=1)
        parent[own[has_denser]] = nearest[has_denser]
        delta[own[has_denser]] = candidates[has_denser, nearest[has_denser]]
        # the densest point gets its distance to the farthest point
        delta[own[~has_denser]] = block[~has_denser].max(axis=1)
    return parent, delta


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"],
    k: int = 5,
    distance: PeakDistance = "weighted_jaccard",
    block_elements: int = DISTANCE_BLOCK_ELEMENTS,
) -> DensityPeakForest:
    """`points` are compared as `distance` sees them: contribution shares, or L2-normalised rows."""
    rows = comparable_rows(points.astype(np.float64), distance)
    # rounding lets float noise still count as a copy
    # distinct_of_hit[h] is the row of `distinct` that hit h became, counts[p] is how many hits became row p
    distinct, distinct_of_hit, counts = np.unique(rows.round(6), axis=0, return_inverse=True, return_counts=True)
    kth_distance = _kth_distances(distinct, counts, k, distance, block_elements)
    ranks = _density_ranks(kth_distance, counts)
    parent, delta = _nearest_denser(distinct, ranks, distance, block_elements)
    return DensityPeakForest(
        distinct_of_hit=distinct_of_hit.ravel(), counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta,
    )


def root_mask(
    forest: DensityPeakForest, root_delta: float, root_density_quantile: float
) -> Bool[np.ndarray, "n_distinct"]:
    """Roots are at least `root_delta` from anything denser, and in the densest `root_density_quantile` of points.

    - The density cut keeps out outliers: they are far from everything, but sparse.
    - The densest point has no parent, so it is always a root."""
    dense_enough = forest.kth_distance <= np.quantile(forest.kth_distance, root_density_quantile)
    return ((forest.delta >= root_delta) & dense_enough) | (forest.parent < 0)


def forest_trees(forest: DensityPeakForest, root_delta: float, root_density_quantile: float) -> ForestTrees:
    """Points are visited densest first, so a point's parent always has its tree and depth already."""
    is_root = root_mask(forest, root_delta, root_density_quantile)
    root_of = np.empty(len(is_root), dtype=int)
    depth = np.zeros(len(is_root), dtype=int)
    for point in np.argsort(-forest.density_rank):
        if is_root[point]:
            root_of[point] = point
        else:
            root_of[point], depth[point] = root_of[forest.parent[point]], depth[forest.parent[point]] + 1
    roots = np.flatnonzero(is_root)
    hits_per_root = np.array([forest.counts[root_of == r].sum() for r in roots])
    roots = roots[np.argsort(-hits_per_root, kind="stable")]
    tree = np.empty(len(is_root), dtype=int)
    tree[roots] = np.arange(len(roots))
    tree = tree[root_of]
    return ForestTrees(roots=roots, tree=tree, depth=depth)

