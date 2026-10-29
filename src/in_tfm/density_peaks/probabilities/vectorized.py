"""The implementation callers use. `naive.py` computes the same things one point at a time."""

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
    cores = []
    for tree, root in enumerate(trees.roots):
        points = np.flatnonzero(trees.tree == tree)
        cores.append(points[forest.kth_distance[points] <= forest.kth_distance[root] / peak_share])
    return cores


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
