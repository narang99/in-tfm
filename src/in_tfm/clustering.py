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


class HDBSCANBackend(Protocol):
    """The slice of the HDBSCAN API used here - cuml and sklearn-contrib hdbscan agree on it."""

    def __init__(self, min_cluster_size: int, cluster_selection_method: str) -> None: ...
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


SelectionMethod = Literal["eom", "leaf"]
"""How HDBSCAN picks clusters from its condensed tree.

- `eom` (excess of mass): may pick a large parent over its children, so distinct sub-clusters
  can be merged into one big cluster.
- `leaf`: takes the leaves, giving many smaller and more homogeneous clusters.
"""


def cluster_labels(
    points: Float[np.ndarray, "n_points dim"],
    min_cluster_size: int,
    cluster_selection_method: SelectionMethod = "eom",
) -> Int[np.ndarray, "n_points"]:
    """Cluster ids per point, with HDBSCAN's noise label (-1) left in place for callers to drop."""
    backend, name = resolve_hdbscan_backend()
    print(f"[cluster] {len(points)} points x {points.shape[1]} dims via {name}, {cluster_selection_method}")

    clusterer = backend(min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method)
    clusterer.fit(points)
    return np.asarray(clusterer.labels_)
