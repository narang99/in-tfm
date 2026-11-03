"""The parts every decoder language model adapter shares."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from ..attribution import compute_attnlrp_relevance
from ..presenters import TextPresenter
from ..sources import TextSource
from .base import reject_nnsight_wrapped


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer is None:
        raise ValueError(f"no tokenizer found for {name}")
    return tokenizer


def force_eager_attention(model: torch.nn.Module) -> None:
    """The AttnLRP patches replace the eager kernel, so the model must not pick sdpa or flash."""
    config = model.config
    assert isinstance(config, PretrainedConfig)
    config._attn_implementation = "eager"


class DecoderTextAdapter(ABC):
    """Wraps an HF-shaped causal LM, whose decoder stack is `hf_model.model`."""

    attr_fn = staticmethod(compute_attnlrp_relevance)
    prepend_bos = False

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
        return self.hf_model.get_submodule("model")

    def patch_for_attn_lrp(self) -> None:
        if self._patched:
            return
        reject_nnsight_wrapped(self.get_model())
        self._apply_attn_lrp_patch()
        self._patched = True

    @abstractmethod
    def _apply_attn_lrp_patch(self) -> None: ...

    def make_source(self, samples: Sequence[str]) -> TextSource:
        return TextSource(
            samples,
            self.tokenizer,
            self.get_model().get_submodule("embed_tokens"),
            self.max_length,
            self.prepend_bos,
        )

    def make_presenter(self, source: TextSource, clustered_label: str) -> TextPresenter:
        return TextPresenter(source, clustered_label=clustered_label)
