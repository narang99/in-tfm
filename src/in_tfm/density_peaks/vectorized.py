"""The forest with numpy, a block of rows at a time. `naive.py` computes the same with plain loops."""

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.spatial.distance import cdist

from ..weighted_jaccard import weighted_jaccard_block
from .forest import (
    DensityPeakForest, ForestTrees, PeakDistance, RootCuts, comparable_rows, merge_copies, root_cuts, root_mask,
)

DISTANCE_BLOCK_ELEMENTS = 2**26
"""The default cap on a block of rows against every point, about 0.5 GB of float64.

- Tests pass a small value to force several blocks."""


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


def kth_distances(
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


def nearest_neighbours(
    rows: Float[np.ndarray, "n_distinct compared_dim"], n_neighbours: int, distance: PeakDistance, block_elements: int
) -> tuple[Int[np.ndarray, "n_distinct n_neighbours"], Float[np.ndarray, "n_distinct n_neighbours"]]:
    """Returns (neighbours, their distances): other distinct points, closest first, ties to the lower index.

    - A tie at the edge of the kept set can still differ from `naive.py`, since `argpartition` picks one arbitrarily."""
    n = min(n_neighbours, len(rows) - 1)
    neighbours = np.empty((len(rows), max(n, 0)), dtype=int)
    distances = np.empty((len(rows), max(n, 0)))
    if n <= 0:
        return neighbours, distances
    for start, block in _row_blocks(rows, distance, block_elements):
        own = np.arange(start, start + len(block))
        # a point is never its own neighbour
        block[np.arange(len(block)), own] = np.inf
        nearest = np.argpartition(block, n - 1, axis=1)[:, :n]
        nearest_distance = np.take_along_axis(block, nearest, axis=1)
        # lexsort sorts by its last key first: distance, then index
        closest_first = np.lexsort((nearest, nearest_distance), axis=1)
        neighbours[own] = np.take_along_axis(nearest, closest_first, axis=1)
        distances[own] = np.take_along_axis(nearest_distance, closest_first, axis=1)
    return neighbours, distances


def density_ranks(
    kth_distance: Float[np.ndarray, "n_distinct"], counts: Int[np.ndarray, "n_distinct"]
) -> Int[np.ndarray, "n_distinct"]:
    """Ties on distance, common at 0, go to the point with more copies."""
    # lexsort sorts by its last key first: larger kth_distance first, then fewer copies first
    sparsest_first = np.lexsort((counts, -kth_distance))
    # invert the order: the point at position i of sparsest_first gets rank i
    ranks = np.empty(len(kth_distance), dtype=int)
    ranks[sparsest_first] = np.arange(len(kth_distance))
    return ranks


def nearest_denser(
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


def forest_trees(forest: DensityPeakForest, cuts: RootCuts | None = None) -> ForestTrees:
    """`cuts` defaults to `root_cuts`, the elbows.

    - Roots and depths come from pointer jumping, which `root_and_depth` explains."""
    cuts = cuts or root_cuts(forest)
    is_root = root_mask(forest, cuts)
    root_of, depth = root_and_depth(forest.parent, is_root)
    roots = np.flatnonzero(is_root)
    hits_per_root = np.bincount(root_of, weights=forest.counts, minlength=len(is_root))[roots]
    roots = roots[np.argsort(-hits_per_root, kind="stable")]
    # a lookup from root point to tree number, read only at root points
    tree_of_root = np.empty(len(is_root), dtype=int)
    tree_of_root[roots] = np.arange(len(roots))
    tree = tree_of_root[root_of]
    return ForestTrees(cuts=cuts, roots=roots, tree=tree, depth=depth)


def root_and_depth(
    parent: Int[np.ndarray, "n"], is_root: Bool[np.ndarray, "n"]
) -> tuple[Int[np.ndarray, "n"], Int[np.ndarray, "n"]]:
    """Each point's root, and its number of steps up to it.

    - Each round, every point jumps from its current ancestor to that ancestor's ancestor, and adds its hops.
      - A root points at itself with 0 hops, so a point that reached its root stays there.
    - The jump doubles each round, so the rounds grow with log(depth), not with the number of points.
      - The naive version walks densest first instead, one point at a time."""
    own = np.arange(len(parent))
    assert (parent[~is_root] >= 0).all(), "only a root can be without a parent"
    up = np.where(is_root, own, parent)
    hops = (~is_root).astype(int)
    while True:
        next_up = up[up]
        if np.array_equal(next_up, up):
            return up, hops
        hops, up = hops + hops[up], next_up


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"],
    k: int = 5,
    distance: PeakDistance = "weighted_jaccard",
    normalise: bool = True,
    block_elements: int = DISTANCE_BLOCK_ELEMENTS,
) -> DensityPeakForest:
    """`points` are compared as `comparable_rows` makes them."""
    distinct, distinct_of_hit, counts = merge_copies(comparable_rows(points.astype(np.float64), distance, normalise))
    kth_distance = kth_distances(distinct, counts, k, distance, block_elements)
    ranks = density_ranks(kth_distance, counts)
    parent, delta = nearest_denser(distinct, ranks, distance, block_elements)
    neighbours, neighbour_distance = nearest_neighbours(distinct, k, distance, block_elements)
    return DensityPeakForest(
        distinct_of_hit=distinct_of_hit, counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta, neighbours=neighbours, neighbour_distance=neighbour_distance,
    )
