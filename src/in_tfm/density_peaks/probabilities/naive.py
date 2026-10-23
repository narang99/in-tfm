"""The plain-loop mirror of `vectorized.py`: the same functions, one point and one edge at a time."""

import heapq
import math
import statistics

import numpy as np
from jaxtyping import Float, Int

from ..forest import DensityPeakForest, ForestTrees

Graph = list[dict[int, float]]
"""Per point, its neighbours and the edge length to each. The vectorized version holds the same edges in a sparse matrix."""


def calculate_probs(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    return geodesic_probabilities(forest, trees, peak_share)


def geodesic_probabilities(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    distances = geodesic_distances(knn_graph(forest), peak_cores(forest, trees, peak_share))
    probabilities = softmax_over_trees(distances, step_length(forest))
    for point, row in enumerate(distances):
        if not any(math.isfinite(d) for d in row):
            probabilities[point] = 0.0
            probabilities[point, trees.tree[point]] = 1.0
    return probabilities


def peak_cores(forest: DensityPeakForest, trees: ForestTrees, peak_share: float) -> list[Int[np.ndarray, "n_peak"]]:
    cores = []
    for tree, root in enumerate(trees.roots):
        cut = forest.kth_distance[root] / peak_share
        core = [point for point in range(len(trees.tree))
                if trees.tree[point] == tree and forest.kth_distance[point] <= cut]
        cores.append(np.array(core, dtype=int))
    return cores


def knn_graph(forest: DensityPeakForest) -> Graph:
    """An edge goes both ways. When both ends list each other, the longer length wins, as `maximum` does in the sparse version."""
    n, k = forest.neighbours.shape
    graph: Graph = [{} for _ in range(n)]
    for point in range(n):
        for j in range(k):
            other = int(forest.neighbours[point, j])
            length = max(float(forest.neighbour_distance[point, j]), np.finfo(float).tiny)
            for a, b in ((point, other), (other, point)):
                graph[a][b] = max(graph[a].get(b, 0.0), length)
    return graph


def geodesic_distances(graph: Graph, cores: list[np.ndarray]) -> Float[np.ndarray, "n_distinct n_trees"]:
    """One Dijkstra per core, started from all its points at once at distance 0."""
    distances = np.full((len(graph), len(cores)), np.inf)
    for tree, core in enumerate(cores):
        queue = [(0.0, int(point)) for point in core]
        heapq.heapify(queue)
        while queue:
            distance, point = heapq.heappop(queue)
            if distance >= distances[point, tree]:
                continue
            distances[point, tree] = distance
            for other, length in graph[point].items():
                if distance + length < distances[other, tree]:
                    heapq.heappush(queue, (distance + length, other))
    return distances


def step_length(forest: DensityPeakForest) -> float:
    nearest = [float(row[0]) for row in forest.neighbour_distance if len(row)]
    usable = [d for d in nearest if math.isfinite(d) and d > 0]
    return statistics.median(usable) if usable else 1.0


def softmax_over_trees(distances: Float[np.ndarray, "n n_trees"], tau: float) -> Float[np.ndarray, "n n_trees"]:
    probabilities = np.zeros_like(distances)
    for point, row in enumerate(distances):
        finite = [d for d in row if math.isfinite(d)]
        nearest = min(finite) if finite else 0.0
        weights = [math.exp(-(d - nearest) / tau) for d in row]
        total = sum(weights)
        if total > 0:
            probabilities[point] = [w / total for w in weights]
    return probabilities
