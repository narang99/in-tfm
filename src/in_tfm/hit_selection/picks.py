"""What each strategy returns.

- Strategies see the column already flipped to the tail being explained, so every value here is a magnitude.
  - `HitSelection.threshold` puts the sign back.
"""

from typing import Annotated, Literal

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field


class PickedPositions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    n_candidates: int
    """Positions beyond the cutoff or floor, before the `max_hits` cap."""


class ElbowPick(BaseModel):
    method: Literal["elbow"] = "elbow"
    cutoff: float
    elbow_idx: int
    """Index into `HitSelection.sorted_magnitudes`."""

    @property
    def threshold(self) -> float:
        return self.cutoff


class BucketedPick(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: Literal["bucketed"] = "bucketed"
    floor: float
    bucket_ids: Int[np.ndarray, "n_hits"]
    """Merged bucket per kept hit."""
    bucket_edges: Float[np.ndarray, "n_edges"]
    """Lower edge of every merged bucket, then the max."""

    @property
    def threshold(self) -> float:
        return self.floor


Pick = Annotated[ElbowPick | BucketedPick, Field(discriminator="method")]
