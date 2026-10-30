"""The implementation callers use. `naive.py` computes the same things one point at a time."""

import multiprocessing

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int
from scipy.sparse.csgraph import dijkstra

from ..forest import DensityPeakForest, ForestTrees


def calculate_probs(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    """The measure the rest of the pipeline reads. It is geodesic distance to each tree's peak for now."""
    return geodesic_probabilities(forest, trees, peak_share)


def geodesic_probabilities(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    """A point that reaches no peak along the graph falls back to its own tree, all its probability there."""
    distances = distances_to_peaks(forest, trees, peak_share)
    probabilities = softmax_over_trees(distances, step_length(forest))
    unreachable = ~np.isfinite(distances).any(axis=1)
    probabilities[unreachable] = np.eye(len(trees.roots))[trees.tree[unreachable]]
    return probabilities


def distances_to_peaks(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    return geodesic_distances(knn_graph(forest), peak_cores(forest, trees, peak_share))


def peak_cores(forest: DensityPeakForest, trees: ForestTrees, peak_share: float) -> list[Int[np.ndarray, "n_peak"]]:
    """Each tree's points at least `peak_share` as dense as its root, the tree's densest point.

    - Density is 1 / kth distance, so the cut is a kth distance at most the root's divided by the share.
    - The root always qualifies, so no tree is left without a peak."""
    in_core = forest.kth_distance <= (forest.kth_distance[trees.roots] / peak_share)[trees.tree]
    points = np.flatnonzero(in_core)
    # a stable sort keeps each core's points in index order
    points = points[np.argsort(trees.tree[points], kind="stable")]
    sizes = np.bincount(trees.tree[points], minlength=len(trees.roots))
    return np.split(points, np.cumsum(sizes)[:-1])


def knn_graph(forest: DensityPeakForest) -> sp.csr_matrix:
    """Each point joined to its k nearest, weighted by their distance, read both ways.

    - A zero-distance edge would vanish from a sparse matrix, so it gets the smallest positive float instead."""
    n, k = forest.neighbours.shape
    source = np.repeat(np.arange(n), k)
    weight = np.maximum(forest.neighbour_distance.ravel(), np.finfo(float).tiny)
    directed = sp.csr_matrix((weight, (source, forest.neighbours.ravel())), shape=(n, n))
    return directed.maximum(directed.T)


def geodesic_distances(graph: sp.csr_matrix, cores: list[np.ndarray]) -> Float[np.ndarray, "n_distinct n_trees"]:
    """The shortest path from each point to the closest point of each core. Inf when a core is out of reach."""
    return np.stack([dijkstra(graph, directed=False, indices=core, min_only=True) for core in cores], axis=1)


def geodesic_entropy(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float, margin_steps: float, workers: int = 1,
) -> Float[np.ndarray, "n_distinct"]:
    """The entropy in bits of each point's probabilities over trees, without the points x trees table.

    - Trees more than `margin_steps` steps of τ past a point's nearest peak are left out.
      - Their weight is below exp(-margin_steps) of the nearest tree's, so they barely change the entropy.
    - Each tree's Dijkstra runs on `past_nearest_peak` edges, with `margin_steps` as its limit.
      - A path's length there is how far it runs past the nearest peak, and it never shrinks along the path.
      - So the search stops at the points the tree still counts for, and the cost follows the kept entries.
    - The entropy comes from two running sums, so no point keeps its trees.
      - With e the steps past the nearest peak, a tree's weight is w = exp(-e).
      - Then the entropy in nats is log(sum w) + sum(w e) / sum w.
    - A point out of reach of every peak gets 0, as its own tree gets all of its probability.
    - `workers` splits the trees over forked processes. Each returns its own two sums, and they add up."""
    graph = knn_graph(forest)
    cores = peak_cores(forest, trees, peak_share)
    tau = step_length(forest)
    nearest = dijkstra(graph, directed=False, indices=np.concatenate(cores), min_only=True) / tau
    steps = _past_nearest_peak(graph / tau, nearest)
    total_weight, weighted_steps = _tree_sums(steps, cores, margin_steps, workers)
    nats = np.zeros(len(nearest))
    reachable = total_weight > 0
    nats[reachable] = np.log(total_weight[reachable]) + weighted_steps[reachable] / total_weight[reachable]
    return nats / np.log(2)


_FORKED: tuple[sp.csr_matrix, list[np.ndarray], float, int] | None = None
"""What forked workers read, since a fork shares the parent's memory without pickling the graph."""


def _tree_sums(
    steps: sp.csr_matrix, cores: list[np.ndarray], margin_steps: float, workers: int
) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    global _FORKED
    if workers <= 1:
        return _sums_over(steps, cores, margin_steps)
    _FORKED = (steps, cores, margin_steps, workers)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_forked_sums, range(workers))
    finally:
        _FORKED = None
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def _forked_sums(worker: int) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """Trees are dealt out in turn, so each worker gets large and small trees alike."""
    steps, cores, margin_steps, workers = _FORKED
    return _sums_over(steps, cores[worker::workers], margin_steps)


def _sums_over(
    steps: sp.csr_matrix, cores: list[np.ndarray], margin_steps: float
) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    total_weight = np.zeros(steps.shape[0])
    weighted_steps = np.zeros(steps.shape[0])
    for core in cores:
        past = dijkstra(steps, directed=True, indices=core, min_only=True, limit=margin_steps)
        reached = np.flatnonzero(np.isfinite(past))
        weight = np.exp(-past[reached])
        total_weight[reached] += weight
        weighted_steps[reached] += weight * past[reached]
    return total_weight, weighted_steps


def _past_nearest_peak(graph: sp.csr_matrix, nearest: Float[np.ndarray, "n"]) -> sp.csr_matrix:
    """Each edge u -> v made length + nearest[u] - nearest[v], which is Johnson's reweighting.

    - Nearest-peak distance moves by at most the edge's length, so no edge goes negative.
      - Rounding can still take one just below 0, and sparse matrices drop exact zeros.
      - So a length is kept at least the smallest float.
    - Edges out of points that reach no peak, and edges of infinite length, are dropped.
      - Their new length would be inf - inf."""
    coo = graph.tocoo()
    with np.errstate(invalid="ignore"):
        length = coo.data + nearest[coo.row] - nearest[coo.col]
    keep = np.isfinite(length)
    return sp.csr_matrix(
        (np.maximum(length[keep], np.finfo(float).tiny), (coo.row[keep], coo.col[keep])), shape=graph.shape
    )


def step_length(forest: DensityPeakForest) -> float:
    """The median distance to the nearest other point, over those that are finite and positive. 1 when there are none."""
    nearest = forest.neighbour_distance[:, 0] if forest.neighbour_distance.shape[1] else np.zeros(0)
    usable = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(usable)) if len(usable) else 1.0


def softmax_over_trees(distances: Float[np.ndarray, "n n_trees"], tau: float) -> Float[np.ndarray, "n n_trees"]:
    """Each row shifted by its smallest finite distance first, so a far-away point does not underflow to all zeros."""
    nearest = np.where(np.isfinite(distances), distances, np.inf).min(axis=1, keepdims=True)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0)
    weights = np.exp(-(distances - nearest) / tau)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
