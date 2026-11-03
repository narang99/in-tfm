"""Pixel-space rendering of a cluster's hits."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float, UInt8
from pydantic import BaseModel

from ..html_report import render_template
from ..viz import mk_overlay, render_hadamard_tiles, to_pil
from .artifacts import ClusterArtifacts
from .base import ClusterHit, HadamardShape


class ImageNormalization(Protocol):
    """Per-channel mean and std a model input was normalised with."""

    image_mean: list[float]
    image_std: list[float]


def inv_tfm(
    processor: ImageNormalization, tensor: Float[torch.Tensor, "*batch c h w"]
) -> UInt8[np.ndarray, "h w c"]:
    """Undo the mean/std normalization, for display (not model input)."""
    mean = np.array(processor.image_mean)
    std = np.array(processor.image_std)
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    img = img.transpose(1, 2, 0) * std + mean
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


class ImageClusterView(BaseModel):
    original_url: str
    overlay_url: str
    hadamard_url: str
    n_hits: int
    n_zero_relevance: int
    """Hits whose relevance is zero everywhere, which an overlay alone shows as a plain image."""


class ImagePresenter:
    """Pixel-space overlays."""

    def __init__(self, processor: ImageNormalization, alpha: float = 0.7) -> None:
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
            n_hits=len(hits),
            n_zero_relevance=sum(not h.relevance.any() for h in hits),
        )
        return render_template("image_cluster.html", view=view)

    def _denormalized(self, hit: ClusterHit) -> np.ndarray:
        """The model input with the processor's mean/std undone - what the overlay is blended
        onto, so the two line up pixel for pixel.

        A single-channel image is repeated to RGB, since PIL cannot build an image from (H, W, 1)
        and the overlay is blended onto an RGB heat map."""
        image = inv_tfm(self.processor, hit.model_input)
        return np.repeat(image, 3, axis=-1) if image.shape[-1] == 1 else image

    def _relevance_map(self, hit: ClusterHit) -> torch.Tensor:
        """Collapse the channel axis of a (C, H, W) relevance array down to (H, W)."""
        relevance = hit.relevance
        if relevance.ndim == 4:
            relevance = relevance[0]
        return torch.from_numpy(relevance.sum(0))
