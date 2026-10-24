"""Buckets of hits by which clusters carry their probability.

- A cluster bucket holds the hits where one cluster has at least a given probability, at several tiers.
  - Tiers are nested, so a hit can be in more than one.
- A set bucket holds the hits that need exactly the same clusters to reach `SET_MASS` of their probability.
  - A hit is placed by the smallest number of top clusters that reach the mass, so these buckets do not overlap.
  - A hit that needs more than half the clusters is in no set bucket.
"""

from collections import defaultdict

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

SET_MASS = 0.85
CLUSTER_TIERS = (0.9, 0.8, 0.7)
MASS_TOLERANCE = 1e-6


class Bucket(BaseModel):
    name: str
    clusters: tuple[int, ...]
    hits: list[int]


def clusters_needed(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> Int[np.ndarray, "n_hits"]:
    """How many of the most probable clusters it takes to reach `SET_MASS`."""
    descending = -np.sort(-probabilities, axis=1)
    reached = np.cumsum(descending, axis=1) >= SET_MASS - MASS_TOLERANCE
    return reached.argmax(axis=1) + 1


def cluster_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> list[Bucket]:
    buckets = []
    for cluster in range(probabilities.shape[1]):
        for tier in CLUSTER_TIERS:
            hits = np.flatnonzero(probabilities[:, cluster] >= tier)
            if len(hits):
                buckets.append(Bucket(name=f"cluster {cluster} >= {tier}", clusters=(cluster,), hits=hits.tolist()))
    return buckets


def set_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> list[Bucket]:
    """Sets of 2 clusters up to half of all clusters."""
    n_clusters = probabilities.shape[1]
    needed = clusters_needed(probabilities)
    ranked = np.argsort(-probabilities, axis=1)
    hits_of_set: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for hit in np.flatnonzero((needed >= 2) & (needed <= n_clusters // 2)):
        hits_of_set[tuple(sorted(ranked[hit, : needed[hit]].tolist()))].append(int(hit))
    return [
        Bucket(name="clusters " + " + ".join(map(str, clusters)), clusters=clusters, hits=hits)
        for clusters, hits in hits_of_set.items()
    ]


def all_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> list[Bucket]:
    """Every non-empty bucket, largest first."""
    buckets = cluster_buckets(probabilities) + set_buckets(probabilities)
    return sorted(buckets, key=lambda b: -len(b.hits))
