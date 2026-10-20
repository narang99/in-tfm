"""Density-peak forest over a neuron's hits (Rodriguez and Laio, 2014), with a kNN density.

- Every hit links to its nearest denser hit, so following links always climbs to a peak.
- A root is a hit that is dense and far from anything denser.
  - Both cut-offs are elbows: of the sorted density curve, and of the sorted delta curve.
  - Each root grows one tree, and each tree is a cluster.
- Unlike HDBSCAN, it needs no low-density gap between groups.
  - So it still splits a continuous cloud, where HDBSCAN returns one cluster or mostly noise.
  - The cost: it cannot tell two separate peaks from two bumps on one hill.
- Identical hits are merged first, with a count.
  - An image patch repeats often, for example a fully inked 3x3 patch, and copies would tie on density.
"""

from .forest import (
    DensityPeakForest, ForestTrees, PeakDistance, RootCuts, comparable_rows, elbow, log_density, merge_copies, root_cuts,
    root_mask,
)
from .vectorized import (
    DISTANCE_BLOCK_ELEMENTS, density_peak_forest, density_ranks, forest_trees, kth_distances, nearest_denser,
)

__all__ = [
    "DISTANCE_BLOCK_ELEMENTS", "DensityPeakForest", "ForestTrees", "PeakDistance", "RootCuts", "comparable_rows",
    "density_peak_forest", "density_ranks", "elbow", "forest_trees", "kth_distances", "log_density", "merge_copies",
    "nearest_denser", "root_cuts", "root_mask",
]
