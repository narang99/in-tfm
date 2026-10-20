"""What a cluster and a group of clusters have in common.

- Code that only summarises or compares clusters takes `ClusterLike`, so it works on either level.
- Code that needs what only one level has, such as a cluster's displayed hits, takes the concrete type.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float


@runtime_checkable
class ClusterLike(Protocol):
    @property
    def id(self) -> int:
        """Unique within its level of one report, so a cluster id and a group id can be equal."""
        ...

    @property
    def n_hits(self) -> int: ...

    def mean_hadamard(self) -> Float[torch.Tensor, "hidden"]:
        """The mean over every hit, not only the displayed ones."""
        ...

    def activations(self) -> Float[np.ndarray, "n_hits"]:
        """The signed activation of every hit."""
        ...


def median_activation(unit: ClusterLike) -> float:
    return float(np.median(unit.activations()))


def activations_by_id(units: Sequence[ClusterLike]) -> dict[int, list[float]]:
    """The shape the activation range plot takes."""
    return {unit.id: unit.activations().tolist() for unit in units}


def unit_means(units: Sequence[ClusterLike]) -> Float[torch.Tensor, "units hidden"]:
    """L2-normalised, so a dot product of two rows is their cosine."""
    return torch.nn.functional.normalize(torch.stack([u.mean_hadamard().double() for u in units]), dim=1)
