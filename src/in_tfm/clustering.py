"""HDBSCAN over Hadamard products, on GPU where one is available.

cuml's GPU HDBSCAN is roughly two orders of magnitude faster than the CPU build, which matters
more than it looks: HDBSCAN on vectors this wide (3072) gets no help from its tree acceleration
and degrades to a near-quadratic brute-force comparison, so cost grows with the square of the
hit count. At ~1200 hits the CPU build takes ~10s; at ~6500 it takes ~5 minutes, per neuron.
"""

from functools import cache
from typing import Literal, Protocol

import numpy as np
from jaxtyping import Float, Int
from sklearn.preprocessing import normalize


ClusterSelectionMethod = Literal["eom", "leaf"]
ClusteringDistance = Literal["euclidean", "weighted_jaccard"]


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


def contribution_shares(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points two_dim"]:
    """Each row's positive and negative parts side by side, L1-normalised.

    A row says where its contribution comes from, with signs kept apart so +a and -a never match.
    An all-zero row stays all zero."""
    parts = np.concatenate([points.clip(min=0), (-points).clip(min=0)], axis=1).astype(np.float64)
    return normalize(parts, norm="l1")


def weighted_jaccard_block(
    block: Float[np.ndarray, "b two_dim"], shares: Float[np.ndarray, "n_points two_dim"]
) -> Float[np.ndarray, "b n_points"]:
    """Distances from each row of `block` to every row of `shares`, both already contribution shares."""
    overlap = np.minimum(block[:, None, :], shares[None]).sum(axis=-1)
    union = np.maximum(block[:, None, :], shares[None]).sum(axis=-1)
    return 1.0 - overlap / np.where(union > 0, union, 1.0)


def weighted_jaccard_distances(
    points: Float[np.ndarray, "n_points dim"], chunk: int = 256
) -> Float[np.ndarray, "n_points n_points"]:
    """1 - sum(min) / sum(max) between the rows' contribution shares, also called the Ruzicka distance.

    - It is plain Jaccard with each dim partly in a set, so a dim counts in proportion to its size.
    - It is a metric.
    - Pairs are built a chunk of rows at a time, since a full n x n x dim array would be large.
    - The math, line by line, with the chunking and a worked example: https://github.com/narang99/in-tfm/pull/91
    - tests/test_clustering_distance.py checks it against a naive pair-by-pair implementation."""
    shares = contribution_shares(points)
    distances = np.empty((len(shares), len(shares)))
    for start in range(0, len(shares), chunk):
        distances[start:start + chunk] = weighted_jaccard_block(shares[start:start + chunk], shares)
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
) -> Int[np.ndarray, "n_points"]:
    """Cluster ids per point, with HDBSCAN's noise label (-1) left in place for callers to drop.

    `leaf` is the default over HDBSCAN's own `eom`: eom merges toward a few large clusters and
    sends the rest to noise, while leaf keeps the finest clusters in the tree.

    `min_samples=None` leaves HDBSCAN's own default, which is `min_cluster_size`.
    A smaller value makes it less strict about what counts as dense, so fewer points are noise.

    `weighted_jaccard` compares where each point's contribution comes from: see `weighted_jaccard_distances`.
    It always runs on CPU."""
    if distance == "weighted_jaccard":
        print(f"[cluster] {len(points)} points x {points.shape[1]} dims, weighted jaccard, via sklearn (cpu)")
        distances = weighted_jaccard_distances(points)
        return _cluster_precomputed(distances, min_cluster_size, cluster_selection_method, min_samples)

    backend, name = resolve_hdbscan_backend()
    print(f"[cluster] {len(points)} points x {points.shape[1]} dims via {name}")

    clusterer = backend(
        min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method, min_samples=min_samples
    )
    clusterer.fit(points)
    return np.asarray(clusterer.labels_)
