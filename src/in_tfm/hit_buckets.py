"""Buckets of hits by which clusters carry their probability.

- A cluster bucket holds the hits where one cluster has at least a given probability, at several tiers.
  - Tiers are nested, so a hit can be in more than one.
- A set bucket holds the hits that need exactly the same clusters to reach `set_mass` of their probability.
  - A hit is placed by the smallest number of top clusters that reach the mass, so these buckets do not overlap.
  - A hit that needs more than half the clusters is in no set bucket.
"""

import re
from collections import defaultdict

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from .run_config import BucketsConfig

MASS_TOLERANCE = 1e-6


class Bucket(BaseModel):
    name: str
    clusters: tuple[int, ...]
    hits: list[int]

    @property
    def slug(self) -> str:
        """A file-safe name. Tiers stay distinct, since `cluster 0 >= 0.9` becomes `cluster_0_0_9`."""
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


def clusters_needed(probabilities: Float[np.ndarray, "n_hits n_clusters"], set_mass: float) -> Int[np.ndarray, "n_hits"]:
    """How many of the most probable clusters it takes to reach `set_mass`."""
    descending = -np.sort(-probabilities, axis=1)
    reached = np.cumsum(descending, axis=1) >= set_mass - MASS_TOLERANCE
    return reached.argmax(axis=1) + 1


def cluster_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"], tiers: list[float]) -> list[Bucket]:
    buckets = []
    for cluster in range(probabilities.shape[1]):
        for tier in tiers:
            hits = np.flatnonzero(probabilities[:, cluster] >= tier)
            if len(hits):
                buckets.append(Bucket(name=f"cluster {cluster} >= {tier}", clusters=(cluster,), hits=hits.tolist()))
    return buckets


def set_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"], set_mass: float) -> list[Bucket]:
    """Sets of 2 clusters up to half of all clusters."""
    n_clusters = probabilities.shape[1]
    needed = clusters_needed(probabilities, set_mass)
    ranked = np.argsort(-probabilities, axis=1)
    hits_of_set: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for hit in np.flatnonzero((needed >= 2) & (needed <= n_clusters // 2)):
        hits_of_set[tuple(sorted(ranked[hit, : needed[hit]].tolist()))].append(int(hit))
    return [
        Bucket(name="clusters " + " + ".join(map(str, clusters)), clusters=clusters, hits=hits)
        for clusters, hits in hits_of_set.items()
    ]


def all_buckets(probabilities: Float[np.ndarray, "n_hits n_clusters"], config: BucketsConfig) -> list[Bucket]:
    """Every non-empty bucket, largest first."""
    buckets = cluster_buckets(probabilities, config.cluster_tiers) + set_buckets(probabilities, config.set_mass)
    return sorted(buckets, key=lambda b: -len(b.hits))


def paged_buckets(buckets: list[Bucket], config: BucketsConfig) -> list[Bucket]:
    """The buckets that get a page and are sampled for the map."""
    return [b for b in buckets if len(b.hits) >= config.min_hits]


def buckets_of_cluster(buckets: list[Bucket], cluster: int) -> list[Bucket]:
    return [b for b in buckets if cluster in b.clusters]
