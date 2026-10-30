"""The forest with plain loops, one point and one pair at a time. Mirrors `vectorized.py`."""

import math

import numpy as np
from jaxtyping import Float, Int

from ..weighted_jaccard import naive as weighted_jaccard
from .forest import (
    DensityPeakForest, ForestTrees, PeakDistance, RootCuts, comparable_rows, merge_copies, root_cuts, root_mask,
)


def _distance_matrix(rows: Float[np.ndarray, "n_distinct compared_dim"], distance: PeakDistance) -> list[list[float]]:
    if distance == "weighted_jaccard":
        return weighted_jaccard.weighted_jaccard_block(rows, rows).tolist()
    points = rows.tolist()
    return [[math.dist(a, b) for b in points] for a in points]


def kth_distances(
    rows: Float[np.ndarray, "n_distinct compared_dim"],
    counts: Int[np.ndarray, "n_distinct"],
    k: int,
    distance: PeakDistance,
) -> Float[np.ndarray, "n_distinct"]:
    """Each point lists the distance to every other hit, then takes the kth smallest.

    - Another point with 3 copies is listed 3 times.
    - The point's own extra copies are listed at distance 0.
    - With fewer than k other hits, the farthest is used."""
    distances = _distance_matrix(rows, distance)
    kth = []
    for i in range(len(rows)):
        other_hits = [0.0] * (counts[i] - 1)
        for j in range(len(rows)):
            if j != i:
                other_hits += [distances[i][j]] * counts[j]
        other_hits.sort()
        kth.append(other_hits[min(k, len(other_hits)) - 1] if other_hits else 0.0)
    return np.array(kth, dtype=np.float64)


def nearest_neighbours(
    rows: Float[np.ndarray, "n_distinct compared_dim"], n_neighbours: int, distance: PeakDistance
) -> tuple[Int[np.ndarray, "n_distinct n_neighbours"], Float[np.ndarray, "n_distinct n_neighbours"]]:
    """Every other point sorted by distance, then index, and the first `n_neighbours` kept."""
    distances = _distance_matrix(rows, distance)
    n = max(min(n_neighbours, len(rows) - 1), 0)
    neighbours, kept = [], []
    for i in range(len(rows)):
        others = sorted((j for j in range(len(rows)) if j != i), key=lambda j: (distances[i][j], j))[:n]
        neighbours.append(others)
        kept.append([distances[i][j] for j in others])
    return np.array(neighbours, dtype=int).reshape(len(rows), n), np.array(kept, dtype=np.float64).reshape(len(rows), n)


def density_ranks(
    kth_distance: Float[np.ndarray, "n_distinct"], counts: Int[np.ndarray, "n_distinct"]
) -> Int[np.ndarray, "n_distinct"]:
    """Sparsest first: larger kth distance first, then fewer copies first, then lower index first."""
    sparsest_first = sorted(range(len(kth_distance)), key=lambda p: (-kth_distance[p], counts[p]))
    ranks = [0] * len(kth_distance)
    for rank, point in enumerate(sparsest_first):
        ranks[point] = rank
    return np.array(ranks, dtype=int)


def nearest_denser(
    rows: Float[np.ndarray, "n_distinct compared_dim"], ranks: Int[np.ndarray, "n_distinct"], distance: PeakDistance
) -> tuple[Int[np.ndarray, "n_distinct"], Float[np.ndarray, "n_distinct"]]:
    """Ties on distance go to the lower index. The densest point has no parent, and gets its farthest distance."""
    distances = _distance_matrix(rows, distance)
    parents, deltas = [], []
    for i in range(len(rows)):
        parent, delta = -1, math.inf
        for j in range(len(rows)):
            if ranks[j] > ranks[i] and distances[i][j] < delta:
                parent, delta = j, distances[i][j]
        if parent == -1:
            delta = max(distances[i])
        parents.append(parent)
        deltas.append(delta)
    return np.array(parents, dtype=int), np.array(deltas, dtype=np.float64)


def forest_trees(forest: DensityPeakForest, cuts: RootCuts | None = None) -> ForestTrees:
    """`cuts` defaults to `root_cuts`, the elbows.

    Points are visited densest first, so a point's parent always has its tree and depth already."""
    cuts = cuts or root_cuts(forest)
    is_root = root_mask(forest, cuts)
    n = len(is_root)
    root_of, depth = root_and_depth(forest.parent, is_root)

    hits_in_tree = {root: 0 for root in range(n) if is_root[root]}
    for point in range(n):
        hits_in_tree[root_of[point]] += forest.counts[point]
    # largest tree first, ties by lower root index, since sorted keeps the dict's order
    roots = sorted(hits_in_tree, key=lambda root: -hits_in_tree[root])

    tree = [roots.index(root_of[point]) for point in range(n)]
    return ForestTrees(
        cuts=cuts, roots=np.array(roots, dtype=int), tree=np.array(tree, dtype=int), depth=np.array(depth, dtype=int)
    )


def root_and_depth(parent: np.ndarray, is_root: np.ndarray) -> tuple[list[int], list[int]]:
    """Each point walks up its parents until it meets a root, counting the steps."""
    root_of, depth = [], []
    for point in range(len(parent)):
        steps = 0
        while not is_root[point]:
            point, steps = int(parent[point]), steps + 1
        root_of.append(point)
        depth.append(steps)
    return root_of, depth


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"], k: int = 5, distance: PeakDistance = "weighted_jaccard", normalise: bool = True
) -> DensityPeakForest:
    distinct, distinct_of_hit, counts = merge_copies(comparable_rows(points.astype(np.float64), distance, normalise))
    kth_distance = kth_distances(distinct, counts, k, distance)
    ranks = density_ranks(kth_distance, counts)
    parent, delta = nearest_denser(distinct, ranks, distance)
    neighbours, neighbour_distance = nearest_neighbours(distinct, k, distance)
    return DensityPeakForest(
        distinct_of_hit=distinct_of_hit, counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta, neighbours=neighbours, neighbour_distance=neighbour_distance,
    )
