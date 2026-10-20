"""K-means over the unit cluster means, with k picked by the elbow of the inertia curve when not given."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Int
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ...cluster_units import ClusterLike, unit_means
from ...report_meta import KMeansSweep

N_INIT = 20


def elbow_index(values: list[float]) -> int:
    """The point farthest from the straight line between the first and last points, after scaling both axes to [0, 1]."""
    points = np.stack([np.arange(len(values)), values], axis=1).astype(float)
    span = points[-1] - points[0]
    span[span == 0] = 1.0
    points = (points - points[0]) / span
    return int(np.argmax(np.abs(points[:, 1] - points[:, 0])))


def fit(x: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(k, n_init=N_INIT, random_state=seed).fit(x)


def sweep_k(x: np.ndarray, seed: int) -> KMeansSweep:
    """Every k from 1 to one less than the number of clusters, the most that can still merge two."""
    ks = list(range(1, len(x)))
    fits = {k: fit(x, k, seed) for k in ks}
    return KMeansSweep(
        ks=ks,
        inertia=[float(fits[k].inertia_) for k in ks],
        silhouette={k: float(silhouette_score(x, fits[k].labels_)) for k in ks if k >= 2},
    )


def kmeans_assignment(
    units: Sequence[ClusterLike], k: int | None, seed: int
) -> tuple[Int[np.ndarray, "units"], int, KMeansSweep | None]:
    x = unit_means(units).numpy()
    sweep = None
    if k is None:
        sweep = sweep_k(x, seed)
        k = sweep.ks[elbow_index(sweep.inertia)]
    return fit(x, k, seed).labels_, k, sweep
