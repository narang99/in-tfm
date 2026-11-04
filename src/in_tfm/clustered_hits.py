"""Every hit a neuron report clustered, with the cluster it landed in.

- `meta.json` keeps only cluster sizes, and `cluster_{id}/hits.pt` only a sample of each cluster.
- A dependency report needs all of a cluster's hits, to read the layers below at each of them.
- So the report keeps each hit's position and label, and not its Hadamard row.
  - The row is recomputed with one gather over these positions, so the file stays small for every model.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from .cluster_membership import NOISE_LABEL
from .sources import SampleId

CLUSTERED_HITS_NAME = "clustered_hits.json"


class ClusteredHits(BaseModel):
    """One entry per hit in every list, in the order the hits were selected."""

    sample_ids: list[SampleId]
    token_idxs: list[int]
    labels: list[int]
    """`NOISE_LABEL` for a hit in no cluster, or too mixed to be shown in one."""
    activations: list[float]
    """Signed, whichever tail the hit came from."""

    def cluster_rows(self, cluster_id: int) -> list[int]:
        return [i for i, label in enumerate(self.labels) if label == cluster_id]

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(set(self.labels) - {NOISE_LABEL})


def clustered_hits(
    sample_ids: list[SampleId],
    batch_idx: Int[np.ndarray, "n_hits"],
    token_idx: Int[np.ndarray, "n_hits"],
    labels: Int[np.ndarray, "n_hits"],
    activations: Float[np.ndarray, "n_hits"],
) -> ClusteredHits:
    """`batch_idx` indexes into `sample_ids`, as a hit selection does."""
    return ClusteredHits(
        sample_ids=[sample_ids[b] for b in batch_idx],
        token_idxs=token_idx.tolist(),
        labels=labels.tolist(),
        activations=activations.tolist(),
    )


def write_clustered_hits(report_dir: Path, hits: ClusteredHits) -> None:
    (report_dir / CLUSTERED_HITS_NAME).write_text(hits.model_dump_json())


def load_clustered_hits(report_dir: Path) -> ClusteredHits:
    path = report_dir / CLUSTERED_HITS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the report predates saving every hit, so it needs regenerating"
        )
    return ClusteredHits.model_validate_json(path.read_text())
