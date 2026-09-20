"""Turning a cluster's sampled hits into report artifacts.

The other modality-aware module besides `sources`. A presenter owns two decisions that differ
completely between images and text:

- how to collapse the attribution's feature axis (channels for an image, hidden for a token)
- what a rendered hit even is - a JPEG overlay, or highlighted text
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from transformers.image_processing_utils import BaseImageProcessor

from .dicom import inv_tfm
from .sources import SampleId
from .viz import mk_overlay, render_hadamard_tiles, save_image_grid, to_pil


class ClusterHit(NamedTuple):
    """One sampled member of a cluster, with everything needed to render it."""

    sample_id: SampleId
    relevance: Float[np.ndarray, "..."]
    """Unreduced, straight from compute_attnlrp_relevance - the presenter reduces it."""
    model_input: torch.Tensor
    """The leaf that produced it, for presenters that show the input alongside the overlay."""
    hadamard: Float[np.ndarray, "hidden"]


@runtime_checkable
class ClusterPresenter(Protocol):
    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        """Writes artifact files into cluster_dir; returns the markdown fragment linking them."""
        ...


class ImagePresenter:
    """Pixel-space overlays, as in the original DICOM reports."""

    def __init__(
        self, processor: BaseImageProcessor, act_shape: tuple[int, int], alpha: float = 0.7
    ) -> None:
        self.processor = processor
        self.act_shape = act_shape
        self.alpha = alpha

    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        originals = [to_pil(self._denormalized(h), (500, 500)) for h in hits]
        overlays = [
            mk_overlay(self._denormalized(h), self._relevance_map(h), (500, 500), alpha=self.alpha)
            for h in hits
        ]
        tiles = render_hadamard_tiles([h.hadamard.reshape(self.act_shape) for h in hits])

        save_image_grid(originals, cluster_dir / "original.jpg")
        save_image_grid(overlays, cluster_dir / "overlay.jpg")
        save_image_grid(tiles, cluster_dir / "hadamard.jpg")

        name = cluster_dir.name
        return (
            f"![original]({name}/original.jpg)\n\n"
            f"![overlay]({name}/overlay.jpg)\n\n"
            f"<details>\n<summary>hadamard patterns</summary>\n\n"
            f"![hadamard]({name}/hadamard.jpg)\n\n"
            f"</details>\n"
        )

    def _denormalized(self, hit: ClusterHit) -> np.ndarray:
        """The model input with the processor's mean/std undone - what the overlay is blended
        onto, so the two line up pixel for pixel."""
        return inv_tfm(self.processor, hit.model_input)

    def _relevance_map(self, hit: ClusterHit) -> torch.Tensor:
        """Collapse the channel axis of a (C, H, W) relevance array down to (H, W)."""
        relevance = hit.relevance
        if relevance.ndim == 4:
            relevance = relevance[0]
        return torch.from_numpy(relevance.sum(0))
