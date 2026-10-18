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
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

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
    """False at padding. All-True for fixed-size inputs like images.

    Text pads every sequence out to a fixed width, so this is what separates a real token from
    a pad. Sniffing for the pad id is not a substitute: a tokenizer without its own pad token
    falls back to eos below, and then the two are the same integer."""

    display_ids: Int[torch.Tensor, "batch seq"] | None = None
    """Token ids exactly as fed, for presenters that cannot recover the input from the leaf.

    Images don't need this - inv_tfm inverts the processor's normalization, so the picture is
    recoverable from pixel_values. The embedding lookup has no usable inverse, so a text
    presenter that wants to show tokens must either carry them or re-tokenize. 
    We prefer to carry instead of re-tokenizing to make sure 
    we are deriving visualisations from source of truth
    """

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
            display_ids=None if self.display_ids is None else self.display_ids.to(device),
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
