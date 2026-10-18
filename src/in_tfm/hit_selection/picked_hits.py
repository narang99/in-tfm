import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict


class PickedHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    n_above_threshold: int
    bucket_ids: Int[np.ndarray, "n_hits"] | None = None
    """Merged bucket per hit, bucketed selection only."""
    bucket_edges: Float[np.ndarray, "n_edges"] | None = None
    """Lower edge of every merged bucket, then the max. Bucketed selection only."""
