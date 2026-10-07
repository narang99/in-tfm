"""The parts every decoder language model adapter shares."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from ..presenters import TextPresenter
from ..sources import TextSource
from .base import reject_nnsight_wrapped


class DecoderTextAdapter(ABC):
    """Wraps an HF-shaped causal LM, whose decoder stack is `hf_model.model`."""

    def __init__(
        self,
        hf_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._patched = False

    def get_model(self) -> torch.nn.Module:
        return self.hf_model.model

    def patch_for_attn_lrp(self) -> None:
        if self._patched:
            return
        reject_nnsight_wrapped(self.get_model())
        self._apply_attn_lrp_patch()
        self._patched = True

    @abstractmethod
    def _apply_attn_lrp_patch(self) -> None: ...

    def make_source(self, samples: Sequence[str]) -> TextSource:
        return TextSource(samples, self.tokenizer, self.get_model().embed_tokens, self.max_length)

    def make_presenter(self, source: TextSource, clustered_label: str) -> TextPresenter:
        return TextPresenter(source, clustered_label=clustered_label)
