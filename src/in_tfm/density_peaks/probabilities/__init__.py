"""A probability per tree for each point, from how far it is from each tree's peak along the kNN graph.

- Distance runs along the kNN graph, so it follows a tree's shape and pays for crossing sparse ground.
  - A straight distance to a tree's centre treats the tree as a ball, and its probabilities come out flat.
- A tree's peak is its points nearly as dense as its root, so a slightly odd root does not decide the distances alone.
- Probabilities are a softmax of −distance / τ over trees, with τ one typical step: the median nearest-neighbour distance.
- See AGENTS.md for the naive and vectorized split.
"""

from .vectorized import (
    calculate_probs, distances_to_peaks, geodesic_distances, geodesic_probabilities, knn_graph, peak_cores, softmax_over_trees, step_length,
)

__all__ = [
    "calculate_probs", "distances_to_peaks", "geodesic_distances", "geodesic_probabilities", "knn_graph", "peak_cores", "softmax_over_trees",
    "step_length",
]
