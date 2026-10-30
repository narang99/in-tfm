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
    distances = distances_to_peaks(forest, trees, peak_share)
    probabilities = softmax_over_trees(distances, step_length(forest))
    for point, row in enumerate(distances):
        if not any(math.isfinite(d) for d in row):
            probabilities[point] = 0.0
            probabilities[point, trees.tree[point]] = 1.0
    return probabilities


def distances_to_peaks(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> Float[np.ndarray, "n_distinct n_trees"]:
    return geodesic_distances(knn_graph(forest), peak_cores(forest, trees, peak_share))


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
    """A tree's weight is exp(-distance / tau), and its probability is its share of the row's weights.

    - Dividing by tau turns distances into units that are easy to reason about.
      - tau is one typical step: the median nearest-neighbour distance.
      - So distance / tau is the number of typical steps to a tree's peak.
      - One step further means about e times less weight, or 2.7 to 1 odds.
      - Two steps further means about 7 to 1, and three steps about 20 to 1.
    - Without tau, the temperature is 1 in the distance's own units.
      - Weighted Jaccard paths here are tenths, so the weights barely differ.
      - Distances of 0.3 and 0.4 would give 0.53 and 0.47.
      - With tau = 0.05 they are 2 steps apart, and give 0.88 and 0.12.
    - A fixed temperature would not fit every neuron either.
      - The spacing between hits changes from neuron to neuron, and between raw and normalised Hadamards.
      - Taking tau from the data keeps the odds per step the same everywhere.
    - Each row's nearest finite distance is subtracted before the exp.
      - It does not change the result.
      - It multiplies every weight in the row by the same factor, which cancels in weight / total.
      - It keeps the weights from vanishing.
        - exp(-1000) rounds to 0, and a row of zeros would give 0 / 0.
        - After the shift the nearest tree's weight is exactly 1.
    - An infinite distance means the tree's peak is out of reach along the graph.
      - Its weight is exp(-inf) = 0, so it gets no probability.
      - The nearest is taken over finite distances only, since inf - inf is NaN.
      - A row with no finite distance stays all zeros.
        - `geodesic_probabilities` then gives that point to its own tree.
    """
    probabilities = np.zeros_like(distances)
    for point, row in enumerate(distances):
        finite = [d for d in row if math.isfinite(d)]
        nearest = min(finite) if finite else 0.0
        weights = [math.exp(-(d - nearest) / tau) for d in row]
        total = sum(weights)
        if total > 0:
            probabilities[point] = [w / total for w in weights]
    return probabilities


def geodesic_entropy(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float, margin_steps: float
) -> Float[np.ndarray, "n_distinct"]:
    """Trees more than `margin_steps` steps past a point's nearest peak are left out, as in the vectorized version."""
    distances = distances_to_peaks(forest, trees, peak_share)
    tau = step_length(forest)
    entropy = np.zeros(len(distances))
    for point, row in enumerate(distances):
        finite = [d for d in row if math.isfinite(d)]
        if not finite:
            continue
        kept = [d if d - min(finite) <= margin_steps * tau else np.inf for d in row]
        probabilities = softmax_over_trees(np.array([kept]), tau)[0]
        entropy[point] = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy
