"""Two clusters share a group only if they agree both in direction and in size.

- Direction is the cosine of their means, which must be at least `min_cosine`.
  - Cosine alone is lenient: a second component at r of the peak only lowers it by about r^2 / 2.
- Size is the gap between their median activations, which must be at most `max_median_gap`.
  - The gap is a fraction of the largest cluster median, so it follows each neuron's own scale.
  - Cosine cannot see size, since every Hadamard vector is L2-normalised before clustering.
"""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Int
from sklearn.cluster import AgglomerativeClustering

from ...cluster_units import ClusterLike, median_activation, unit_means


def pair_distances(
    units: Sequence[ClusterLike], min_cosine: float, max_median_gap: float
) -> Float[np.ndarray, "units units"]:
    """The larger of the two gaps, each divided by its threshold, so at most 1 means both thresholds hold."""
    means = unit_means(units).numpy()
    medians = np.array([median_activation(u) for u in units])
    direction_gap = (1 - means @ means.T) / (1 - min_cosine)
    size_gap = np.abs(medians[:, None] - medians[None, :]) / np.abs(medians).max() / max_median_gap
    distance = np.maximum(direction_gap, size_gap)
    np.fill_diagonal(distance, 0)
    return distance


def threshold_assignment(
    units: Sequence[ClusterLike], min_cosine: float, max_median_gap: float
) -> Int[np.ndarray, "units"]:
    """Complete linkage merges two groups only while every pair across them is within 1.

    - Single linkage would chain A to B to C even when A and C are far apart.
    """
    if len(units) < 2:
        return np.zeros(len(units), dtype=int)
    linkage = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="complete", distance_threshold=1 + 1e-9
    )
    return linkage.fit(pair_distances(units, min_cosine, max_median_gap)).labels_
