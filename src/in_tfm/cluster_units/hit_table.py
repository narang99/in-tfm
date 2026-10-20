"""Every clustered hit of a report, one row each, so a group can count what its clusters share.

- A cluster's distinct sample count cannot be summed into a group's, since clusters can share samples.
- Only positions, labels and activations are stored.
  - The Hadamard vectors are not, since they would be large for wide layers.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

HIT_TABLE_NAME = "hit_table.npz"


class HitTable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    """Index into the report's `sample_ids`."""
    token_idx: Int[np.ndarray, "n_hits"]
    labels: Int[np.ndarray, "n_hits"]
    """HDBSCAN's label, -1 for noise."""
    activations: Float[np.ndarray, "n_hits"]

    def unique_samples(self, cluster_ids: list[int]) -> int:
        return len(np.unique(self.batch_idx[np.isin(self.labels, cluster_ids)]))


def write_hit_table(report_dir: Path, table: HitTable) -> None:
    np.savez_compressed(report_dir / HIT_TABLE_NAME, **table.model_dump())


def read_hit_table(report_dir: Path) -> HitTable | None:
    """Reports written before the table existed have none."""
    path = report_dir / HIT_TABLE_NAME
    if not path.exists():
        return None
    with np.load(path) as saved:
        return HitTable(**{name: saved[name] for name in saved.files})
