"""A report's clusters, and groups of them, both read back from the report folder.

- `Cluster` is what HDBSCAN found, with everything the report stored for it.
- `ClusterGroup` aggregates its clusters.
  - Its mean is weighted by hits, so it equals the mean over all of the group's hits.
  - Its activations are its clusters' activations, concatenated.
- Without a grouping, every cluster is a group of its own, see `singleton_groups`.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

from .hit_table import HitTable


class MissingActivationsError(ValueError):
    pass


class Cluster(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int
    n_hits: int
    n_unique_samples: int
    mean: Float[torch.Tensor, "hidden"]
    hit_activations: Float[np.ndarray, "n_hits"] | None
    """None when the report has no `activation_ranges.json`."""
    cluster_dir: Path

    def mean_hadamard(self) -> torch.Tensor:
        return self.mean

    def activations(self) -> np.ndarray:
        if self.hit_activations is None:
            raise MissingActivationsError(f"{self.cluster_dir.parent} has no activation_ranges.json")
        return self.hit_activations


class ClusterGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int
    clusters: list[Cluster]
    hit_table: HitTable | None = None
    """Needed for `n_unique_samples`, since clusters can share samples."""

    @property
    def cluster_ids(self) -> list[int]:
        return [c.id for c in self.clusters]

    @property
    def n_hits(self) -> int:
        return sum(c.n_hits for c in self.clusters)

    @property
    def n_unique_samples(self) -> int | None:
        """None for a report written before the hit table existed, unless the group is one cluster."""
        if len(self.clusters) == 1:
            return self.clusters[0].n_unique_samples
        return None if self.hit_table is None else self.hit_table.unique_samples(self.cluster_ids)

    def mean_hadamard(self) -> torch.Tensor:
        weights = torch.tensor([c.n_hits for c in self.clusters], dtype=torch.float64)
        means = torch.stack([c.mean_hadamard().double() for c in self.clusters])
        return (weights[:, None] * means).sum(dim=0) / weights.sum()

    def activations(self) -> np.ndarray:
        return np.concatenate([c.activations() for c in self.clusters])

    def member_cosines(self) -> Float[torch.Tensor, "clusters"]:
        """How far each cluster is from the group's mean, to tell a tight group from a loose one."""
        means = torch.stack([c.mean_hadamard().double() for c in self.clusters])
        return torch.nn.functional.cosine_similarity(means, self.mean_hadamard()[None, :], dim=1)


def singleton_groups(clusters: list[Cluster], hit_table: HitTable | None = None) -> list[ClusterGroup]:
    """Group ids follow the order of `clusters`, which in a report is largest first."""
    return [ClusterGroup(id=i, clusters=[c], hit_table=hit_table) for i, c in enumerate(clusters)]
