"""Pixel-space rendering of a cluster's hits."""

from collections.abc import Sequence

import numpy as np
import torch
from pydantic import BaseModel
from transformers.image_processing_utils import BaseImageProcessor

from ..dicom import inv_tfm
from ..html_report import render_template
from ..viz import mk_overlay, render_hadamard_tiles, to_pil
from .artifacts import ClusterArtifacts
from .base import ClusterHit, HadamardShape


class ImageClusterView(BaseModel):
    original_url: str
    overlay_url: str
    hadamard_url: str


class ImagePresenter:
    """Pixel-space overlays, as in the original DICOM reports."""

    def __init__(self, processor: BaseImageProcessor, alpha: float = 0.7) -> None:
        self.processor = processor
        self.alpha = alpha

    def page_controls(self) -> str:
        return ""

    def render(
        self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape
    ) -> str:
        originals = [to_pil(self._denormalized(h), (500, 500)) for h in hits]
        overlays = [
            mk_overlay(self._denormalized(h), self._relevance_map(h), (500, 500), alpha=self.alpha)
            for h in hits
        ]
        tiles = render_hadamard_tiles([h.hadamard.reshape(hadamard_shape) for h in hits])
        view = ImageClusterView(
            original_url=artifacts.save_grid(originals, "original.jpg"),
            overlay_url=artifacts.save_grid(overlays, "overlay.jpg"),
            hadamard_url=artifacts.save_grid(tiles, "hadamard.jpg"),
        )
        return render_template("image_cluster.html", view=view)

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
