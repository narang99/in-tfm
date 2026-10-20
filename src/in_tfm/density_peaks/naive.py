"""The forest with plain loops, one point and one pair at a time. Mirrors `vectorized.py`."""

import math

import numpy as np
from jaxtyping import Float, Int

from ..weighted_jaccard import naive as weighted_jaccard
from .forest import DensityPeakForest, ForestTrees, PeakDistance, comparable_rows, merge_copies, root_mask


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


def forest_trees(forest: DensityPeakForest, root_delta: float, root_density_quantile: float) -> ForestTrees:
    """Points are visited densest first, so a point's parent always has its tree and depth already."""
    is_root = root_mask(forest, root_delta, root_density_quantile)
    n = len(is_root)
    root_of, depth = [0] * n, [0] * n
    for point in sorted(range(n), key=lambda p: -forest.density_rank[p]):
        if is_root[point]:
            root_of[point] = point
        else:
            parent = forest.parent[point]
            root_of[point], depth[point] = root_of[parent], depth[parent] + 1

    hits_in_tree = {root: 0 for root in range(n) if is_root[root]}
    for point in range(n):
        hits_in_tree[root_of[point]] += forest.counts[point]
    # largest tree first, ties by lower root index, since sorted keeps the dict's order
    roots = sorted(hits_in_tree, key=lambda root: -hits_in_tree[root])

    tree = [roots.index(root_of[point]) for point in range(n)]
    return ForestTrees(
        roots=np.array(roots, dtype=int), tree=np.array(tree, dtype=int), depth=np.array(depth, dtype=int)
    )


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"], k: int = 5, distance: PeakDistance = "weighted_jaccard", normalise: bool = True
) -> DensityPeakForest:
    distinct, distinct_of_hit, counts = merge_copies(comparable_rows(points.astype(np.float64), distance, normalise))
    kth_distance = kth_distances(distinct, counts, k, distance)
    ranks = density_ranks(kth_distance, counts)
    parent, delta = nearest_denser(distinct, ranks, distance)
    return DensityPeakForest(
        distinct_of_hit=distinct_of_hit, counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta,
    )
