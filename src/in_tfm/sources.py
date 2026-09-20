"""Where samples come from, and how they become model inputs.

Everything downstream of activation capture (thresholding, Hadamard products, clustering)
works on `(batch, seq, hidden)` tensors and knows nothing about the modality. This module is
one of the two places that does - the other is `presenters`.

Two things make this more than "hand back a batch dict":

- The gradient leaf is not always the model input. `pixel_values` is float, so attribution can
  differentiate it directly; `input_ids` are integers and cannot be differentiated at all, so a
  text source has to pass embeddings instead and hand back *that* tensor as the leaf.
- Image batches are dense, text batches are padded. Hit selection runs `argwhere` over the
  whole `(batch, seq)` grid, so without a validity mask a padded batch produces hits on pad
  positions - which cluster happily and mean nothing.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict
from transformers.image_processing_utils import BaseImageProcessor

from .dicom import get_batch

SampleId = str
"""Stable identity for one input, carried into meta.json so a hit can be traced back to its
source. A filesystem path for images, something like "wikitext:train:4412" for text."""


class ModelBatch(BaseModel):
    """One batch, in the forms the pipeline needs it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: dict[str, torch.Tensor]
    """Splatted into the model: `model(**kwargs)`."""

    grad_leaf_key: str
    """Which entry of `kwargs` attribution differentiates. For images that is "pixel_values";
    text substitutes "inputs_embeds", since integer input_ids carry no gradient.

    A key rather than the tensor itself: `.to()` copies kwargs and would break any
    identity-based match, silently handing the model a tensor that is not the one gradients
    were requested on."""

    valid_mask: Bool[torch.Tensor, "batch seq"]
    """False at padding. All-True for fixed-size inputs like images."""

    @property
    def grad_leaf(self) -> Float[torch.Tensor, "batch ..."]:
        return self.kwargs[self.grad_leaf_key]

    def differentiable(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Model kwargs with the leaf replaced by a grad-tracking copy, plus that copy."""
        leaf = self.kwargs[self.grad_leaf_key].clone().detach().requires_grad_(True)
        return {**self.kwargs, self.grad_leaf_key: leaf}, leaf

    def to(self, device: str) -> "ModelBatch":
        return ModelBatch(
            kwargs={k: v.to(device) for k, v in self.kwargs.items()},
            grad_leaf_key=self.grad_leaf_key,
            valid_mask=self.valid_mask.to(device),
        )


@runtime_checkable
class SampleSource(Protocol):
    """A corpus, plus how to turn a slice of it into model inputs."""

    def sample_ids(self) -> Sequence[SampleId]:
        """Ordered, and stably so. Slicing the first N must select the same N everywhere -
        filesystem order differs between machines and silently changes which samples an
        experiment ran on."""
        ...

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch: ...


class DicomSource:
    """DICOM files on disk, through a transformers image processor."""

    def __init__(self, dcm_dir: Path, processor: BaseImageProcessor) -> None:
        self.dcm_dir = dcm_dir
        self.processor = processor

    def sample_ids(self) -> Sequence[SampleId]:
        return [str(p) for p in sorted(self.dcm_dir.glob("*.dcm"))]

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        batch = get_batch(self.processor, [Path(i) for i in ids])
        n_images = batch["pixel_values"].shape[0]
        return ModelBatch(
            kwargs=dict(batch),
            # the image itself is the leaf - pixel-space relevance is the whole point of the
            # overlay, so there is nothing to substitute here
            grad_leaf_key="pixel_values",
            valid_mask=torch.ones(n_images, self._n_tokens(), dtype=torch.bool),
        )

    def _n_tokens(self) -> int:
        """Patch grid plus the CLS token - every image yields exactly this many, which is why
        the mask is a formality here."""
        crop = self.processor.crop_size["height"]
        patch = getattr(self.processor, "patch_size", 14)
        return (crop // patch) ** 2 + 1
