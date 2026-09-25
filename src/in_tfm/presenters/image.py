"""Pixel-space rendering of a cluster's hits."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor

from ..dicom import inv_tfm
from ..html_report import details
from ..viz import mk_overlay, render_hadamard_tiles, save_image_grid, to_pil
from .base import ClusterHit, HadamardShape


class ImagePresenter:
    """Pixel-space overlays, as in the original DICOM reports."""

    def __init__(self, processor: BaseImageProcessor, alpha: float = 0.7) -> None:
        self.processor = processor
        self.alpha = alpha

    def render(
        self, hits: Sequence[ClusterHit], cluster_dir: Path, hadamard_shape: HadamardShape
    ) -> str:
        originals = [to_pil(self._denormalized(h), (500, 500)) for h in hits]
        overlays = [
            mk_overlay(self._denormalized(h), self._relevance_map(h), (500, 500), alpha=self.alpha)
            for h in hits
        ]
        tiles = render_hadamard_tiles([h.hadamard.reshape(hadamard_shape) for h in hits])

        save_image_grid(originals, cluster_dir / "original.jpg")
        save_image_grid(overlays, cluster_dir / "overlay.jpg")
        save_image_grid(tiles, cluster_dir / "hadamard.jpg")

        name = cluster_dir.name
        return (
            f'<img src="{name}/original.jpg" alt="original">\n'
            f'<img src="{name}/overlay.jpg" alt="overlay">\n'
            + details("hadamard patterns", f'<img src="{name}/hadamard.jpg" alt="hadamard">')
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
