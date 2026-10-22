"""What a clustering strategy says about each hit: noise, or a probability for every cluster.

- A hard strategy gives one-hot probabilities, so every strategy fits the same contract.
- The report still needs one cluster per hit, so it takes the most probable one, and noise stays noise.
"""

from typing import Annotated, Literal

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field

NOISE_LABEL = -1
"""The label the report reads as noise, and leaves out."""


class Noise(BaseModel):
    kind: Literal["noise"] = "noise"


class ClusterProbabilities(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["probabilities"] = "probabilities"
    probabilities: Float[np.ndarray, "n_clusters"]
    """Sums to 1. Index i is cluster i."""


HitMembership = Annotated[Noise | ClusterProbabilities, Field(discriminator="kind")]


def one_hot_memberships(labels: Int[np.ndarray, "n_hits"]) -> list[HitMembership]:
    """`NOISE_LABEL` becomes `Noise`. Every other label becomes a one-hot vector over all the clusters."""
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    eye = np.eye(n_clusters)
    return [Noise() if label == NOISE_LABEL else ClusterProbabilities(probabilities=eye[label]) for label in labels]


def most_probable_labels(memberships: list[HitMembership]) -> Int[np.ndarray, "n_hits"]:
    """`Noise` becomes `NOISE_LABEL`. Ties go to the lower cluster index."""
    return np.array(
        [NOISE_LABEL if isinstance(m, Noise) else int(np.argmax(m.probabilities)) for m in memberships], dtype=int
    )


def membership_entropy(memberships: list[HitMembership]) -> Float[np.ndarray, "n_hits"]:
    """In bits. 0 is one cluster only, 1 an even split between two. Noise has no clusters, so it gets 0."""
    entropy = np.zeros(len(memberships))
    for i, m in enumerate(memberships):
        if isinstance(m, ClusterProbabilities):
            p = m.probabilities[m.probabilities > 0]
            entropy[i] = float(-(p * np.log2(p)).sum())
    return entropy


def confident_labels(memberships: list[HitMembership], max_entropy: float | None) -> Int[np.ndarray, "n_hits"]:
    """The most probable cluster of each hit, with hits at or above `max_entropy` bits set to `NOISE_LABEL`.

    - The report leaves `NOISE_LABEL` out, so a hit split between clusters does not blur any one of them.
    - Without a threshold every hit keeps its most probable cluster."""
    labels = most_probable_labels(memberships)
    if max_entropy is not None:
        labels[membership_entropy(memberships) >= max_entropy] = NOISE_LABEL
    return labels
