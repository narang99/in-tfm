"""HDBSCAN over Hadamard products, on GPU where one is available.

cuml's GPU HDBSCAN is roughly two orders of magnitude faster than the CPU build, which matters
more than it looks: HDBSCAN on vectors this wide (3072) gets no help from its tree acceleration
and degrades to a near-quadratic brute-force comparison, so cost grows with the square of the
hit count. At ~1200 hits the CPU build takes ~10s; at ~6500 it takes ~5 minutes, per neuron.
"""

from functools import cache
from typing import Literal, Protocol

import numpy as np
from jaxtyping import Bool, Float, Int


ClusterSelectionMethod = Literal["eom", "leaf"]
ClusteringDistance = Literal["euclidean", "mass_overlap"]


class HDBSCANBackend(Protocol):
    """The slice of the HDBSCAN API used here - cuml and sklearn-contrib hdbscan agree on it."""

    def __init__(
        self, min_cluster_size: int, cluster_selection_method: ClusterSelectionMethod, min_samples: int | None
    ) -> None: ...
    def fit(self, points: Float[np.ndarray, "n_points dim"]) -> object: ...

    labels_: Int[np.ndarray, "n_points"]


@cache
def resolve_hdbscan_backend() -> tuple[type[HDBSCANBackend], str]:
    """Returns cuml's GPU HDBSCAN when importable, else the CPU build.

    The CPU package is imported only in the fallback branch, never alongside cuml. The two ship
    overlapping compiled extensions and are known to break each other's API when both are
    loaded, so on a GPU runtime `hdbscan` stays unimported entirely.
    """
    try:
        from cuml.cluster import HDBSCAN as CumlHDBSCAN

        return CumlHDBSCAN, "cuml (gpu)"
    except ImportError:
        from hdbscan import HDBSCAN as CpuHDBSCAN

        return CpuHDBSCAN, "hdbscan (cpu)"


def mass_sets(points: Float[np.ndarray, "n_points dim"], mass: float) -> Bool[np.ndarray, "n_points dim"]:
    """Per row, the smallest set of dims, largest first, whose values reach `mass` of the row's total.

    A row with nothing positive gets an empty set."""
    order = np.argsort(-points, axis=1)
    ranked = np.take_along_axis(points, order, axis=1)
    before = np.cumsum(ranked, axis=1) - ranked
    picked_ranked = (ranked > 0) & (before < mass * ranked.clip(min=0).sum(axis=1, keepdims=True))
    picked = np.zeros_like(picked_ranked)
    np.put_along_axis(picked, order, picked_ranked, axis=1)
    return picked


def mass_overlap_scores(points: Float[np.ndarray, "n_points dim"], mass: float) -> Int[np.ndarray, "n_points n_points"]:
    """Dims two rows share among those carrying `mass` of each sign, positive with positive and negative with negative."""
    positive = mass_sets(points.clip(min=0), mass).astype(np.int32)
    negative = mass_sets((-points).clip(min=0), mass).astype(np.int32)
    return positive @ positive.T + negative @ negative.T


def mass_overlap_distances(points: Float[np.ndarray, "n_points dim"], mass: float) -> Float[np.ndarray, "n_points n_points"]:
    """The highest score between two different rows, minus each pair's score.

    - A row's score with itself is its own set size, not the maximum, so the diagonal is set to 0 by hand.
    - This is not a metric: the triangle inequality can fail.
    - The matrix is n² floats, so it suits hit counts in the low thousands, not tens of thousands."""
    scores = mass_overlap_scores(points, mass).astype(np.float64)
    np.fill_diagonal(scores, -np.inf)
    distances = scores.max() - scores
    np.fill_diagonal(distances, 0.0)
    return distances


def _cluster_precomputed(
    distances: Float[np.ndarray, "n_points n_points"],
    min_cluster_size: int,
    cluster_selection_method: ClusterSelectionMethod,
    min_samples: int | None,
) -> Int[np.ndarray, "n_points"]:
    """sklearn's HDBSCAN, which takes a distance matrix and is safe to load beside cuml.

    sklearn counts a point as its own neighbour and the `hdbscan` package does not.
    The +1 keeps `min_samples` meaning the same as on the euclidean path."""
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=(min_samples or min_cluster_size) + 1,
        cluster_selection_method=cluster_selection_method,
        metric="precomputed",
    )
    return clusterer.fit_predict(distances)


def cluster_labels(
    points: Float[np.ndarray, "n_points dim"],
    min_cluster_size: int,
    cluster_selection_method: ClusterSelectionMethod = "leaf",
    min_samples: int | None = None,
    distance: ClusteringDistance = "euclidean",
    mass: float = 0.9,
) -> Int[np.ndarray, "n_points"]:
    """Cluster ids per point, with HDBSCAN's noise label (-1) left in place for callers to drop.

    `leaf` is the default over HDBSCAN's own `eom`: eom merges toward a few large clusters and
    sends the rest to noise, while leaf keeps the finest clusters in the tree.

    `min_samples=None` leaves HDBSCAN's own default, which is `min_cluster_size`.
    A smaller value makes it less strict about what counts as dense, so fewer points are noise.

    `mass_overlap` compares which dims carry each point, not how much: see `mass_overlap_distances`.
    It always runs on CPU."""
    if distance == "mass_overlap":
        print(f"[cluster] {len(points)} points x {points.shape[1]} dims, mass overlap at {mass:.0%}, via sklearn (cpu)")
        distances = mass_overlap_distances(points, mass)
        return _cluster_precomputed(distances, min_cluster_size, cluster_selection_method, min_samples)

    backend, name = resolve_hdbscan_backend()
    print(f"[cluster] {len(points)} points x {points.shape[1]} dims via {name}")

    clusterer = backend(
        min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method, min_samples=min_samples
    )
    clusterer.fit(points)
    return np.asarray(clusterer.labels_)
