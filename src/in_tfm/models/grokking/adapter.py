from collections.abc import Sequence

import torch

from ...attribution import compute_attnlrp_relevance, patch_eager_attention
from ...presenters import GrokPresenter
from .. import base
from . import model as grok_model
from .model import GrokTransformer
from .source import PairSource


def load_model(checkpoint: str | None) -> GrokTransformer:
    model = GrokTransformer()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    return model.eval()


class GrokkingAdapter:
    """Samples are `(a, b)` pairs, the model is the paper's one-layer transformer."""

    attr_fn = staticmethod(compute_attnlrp_relevance)

    def __init__(self, model: GrokTransformer) -> None:
        self.model = model
        self._patched = False

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> "GrokkingAdapter":
        return cls(load_model(checkpoint))

    def get_model(self) -> GrokTransformer:
        return self.model

    def patch_for_attn_lrp(self) -> None:
        """Attention is the only rule needed, the model has no normalization and its one MLP is a ReLU."""
        if self._patched:
            return
        base.reject_nnsight_wrapped(self.model)
        patch_eager_attention(grok_model)
        self._patched = True

    def make_source(self, samples: Sequence[tuple[int, int]]) -> PairSource:
        return PairSource(samples, self.model)

    def make_presenter(self, source: PairSource, clustered_label: str) -> GrokPresenter:
        return GrokPresenter(clustered_label=clustered_label)
