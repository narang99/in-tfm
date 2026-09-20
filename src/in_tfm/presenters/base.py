"""What every presenter receives, and what every presenter must offer.

Its own module rather than the package `__init__` so `image` and `text` can import it without
importing each other through a half-initialized package.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float

from ..sources import SampleId


class ClusterHit(NamedTuple):
    """One sampled member of a cluster, with everything needed to render it."""

    sample_id: SampleId
    token_idx: int
    """Which position in the sequence crossed the threshold - the token a text report
    highlights, and the patch an image overlay is centred on."""
    relevance: Float[np.ndarray, "..."]
    """Unreduced, straight from compute_attnlrp_relevance - the presenter reduces it."""
    model_input: torch.Tensor
    """The leaf that produced it, for presenters that show the input alongside the overlay."""
    hadamard: Float[np.ndarray, "hidden"]
    display_ids: torch.Tensor | None = None
    """Token ids as fed, when the leaf is not invertible - see sources.ModelBatch."""


@runtime_checkable
class ClusterPresenter(Protocol):
    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        """Writes artifact files into cluster_dir; returns the html fragment linking them."""
        ...
