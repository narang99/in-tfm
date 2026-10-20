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

from .clustering import contribution_shares

PeakDistance = Literal["euclidean", "weighted_jaccard"]
DISTANCE_BLOCK_ELEMENTS = 2**26
"""A block of rows against every point is built at once. This caps its size at about 0.5 GB of float64."""


class DensityPeakForest(BaseModel):
    """Per distinct point unless named per hit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    point_of_hit: Int[np.ndarray, "n_hits"]
    counts: Int[np.ndarray, "n_points"]
    """How many hits share each distinct point."""
    kth_distance: Float[np.ndarray, "n_points"]
    """The distance at which the point's neighbours, counted with their copies, reach k. Smaller is denser."""
    density_rank: Int[np.ndarray, "n_points"]
    """0 is the sparsest point."""
    parent: Int[np.ndarray, "n_points"]
    """The nearest denser point, or -1 for the densest point."""
    delta: Float[np.ndarray, "n_points"]
    """The distance to the parent. The densest point gets its distance to the farthest point."""


class ForestTrees(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    roots: Int[np.ndarray, "n_trees"]
    """Distinct points, largest tree first."""
    tree: Int[np.ndarray, "n_points"]
    """Position in `roots` of the point's tree."""
    depth: Int[np.ndarray, "n_points"]
    """Links from the point up to its root."""


def comparable_rows(points: Float[np.ndarray, "n dim"], distance: PeakDistance) -> Float[np.ndarray, "n d"]:
    if distance == "weighted_jaccard":
        return contribution_shares(points)
    return points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)


def _distance_block(
    block: Float[np.ndarray, "b d"], rows: Float[np.ndarray, "n d"], distance: PeakDistance
) -> Float[np.ndarray, "b n"]:
    if distance == "weighted_jaccard":
        overlap = np.minimum(block[:, None, :], rows[None]).sum(axis=-1)
        union = np.maximum(block[:, None, :], rows[None]).sum(axis=-1)
        return 1.0 - overlap / np.where(union > 0, union, 1.0)
    return np.sqrt(np.maximum(((block[:, None, :] - rows[None]) ** 2).sum(axis=-1), 0.0))


def _row_blocks(rows: Float[np.ndarray, "n d"], distance: PeakDistance):
    """Yields (start, distances of rows[start:start + size] to every row)."""
    size = max(1, DISTANCE_BLOCK_ELEMENTS // (len(rows) * rows.shape[1]))
    for start in range(0, len(rows), size):
        yield start, _distance_block(rows[start:start + size], rows, distance)


def _kth_distances(
    rows: Float[np.ndarray, "n d"], counts: Int[np.ndarray, "n"], k: int, distance: PeakDistance
) -> Float[np.ndarray, "n"]:
    """A point's own extra copies sit at distance 0, so a point seen more than k times has distance 0.

    - The k + 1 nearest always reach k copies, since the point itself is among them.
      - Its own distance is forced to -1 first, so a tie at 0 with another point cannot push it out.
    - With fewer than k other hits in total, the farthest neighbour is used."""
    n = len(rows)
    kth = np.empty(n)
    n_nearest = min(k + 1, n)
    for start, block in _row_blocks(rows, distance):
        own = np.arange(start, start + len(block))
        block[np.arange(len(block)), own] = -1.0
        nearest = np.argpartition(block, n_nearest - 1, axis=1)[:, :n_nearest]
        nearest = np.take_along_axis(nearest, np.argsort(np.take_along_axis(block, nearest, 1), 1), 1)
        copies = np.where(nearest == own[:, None], counts[own, None] - 1, counts[nearest])
        reached = np.minimum((np.cumsum(copies, axis=1) < k).sum(axis=1), n_nearest - 1)
        kth[own] = np.maximum(np.take_along_axis(block, nearest, 1)[np.arange(len(block)), reached], 0.0)
    return kth


def _density_ranks(kth_distance: Float[np.ndarray, "n"], counts: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
    """Ties on distance, common at 0, go to the point with more copies."""
    sparsest_first = np.lexsort((counts, -kth_distance))
    ranks = np.empty(len(kth_distance), dtype=int)
    ranks[sparsest_first] = np.arange(len(kth_distance))
    return ranks


def _nearest_denser(
    rows: Float[np.ndarray, "n d"], ranks: Int[np.ndarray, "n"], distance: PeakDistance
) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    parent = np.full(len(rows), -1)
    delta = np.zeros(len(rows))
    for start, block in _row_blocks(rows, distance):
        own = np.arange(start, start + len(block))
        denser = ranks[None, :] > ranks[own, None]
        candidates = np.where(denser, block, np.inf)
        nearest = candidates.argmin(axis=1)
        has_denser = denser.any(axis=1)
        parent[own[has_denser]] = nearest[has_denser]
        delta[own[has_denser]] = candidates[has_denser, nearest[has_denser]]
        delta[own[~has_denser]] = block[~has_denser].max(axis=1)
    return parent, delta


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"], k: int = 5, distance: PeakDistance = "weighted_jaccard"
) -> DensityPeakForest:
    """`points` are compared as `distance` sees them: contribution shares, or L2-normalised rows."""
    rows = comparable_rows(points.astype(np.float64), distance)
    distinct, point_of_hit, counts = np.unique(rows.round(6), axis=0, return_inverse=True, return_counts=True)
    kth_distance = _kth_distances(distinct, counts, k, distance)
    ranks = _density_ranks(kth_distance, counts)
    parent, delta = _nearest_denser(distinct, ranks, distance)
    return DensityPeakForest(
        point_of_hit=point_of_hit.ravel(), counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta,
    )


def root_mask(forest: DensityPeakForest, root_delta: float, root_density_quantile: float) -> Bool[np.ndarray, "n"]:
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

