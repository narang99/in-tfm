from pathlib import Path
from typing import ClassVar

import torch

from ...attribution import patch_eager_attention
from .. import text
from . import model as attn_only_model
from .model import load_attn_only_2l, load_trained_attn_only


def patch_attn_only_for_attn_lrp(model: torch.nn.Module) -> None:
    """Attention is the only rule needed.

    - The model has no normalization and no MLP, so softmax is its only nonlinearity.
    - Everything else is linear in its input, where plain gradient*input is a valid LRP rule.
    - The patch is scoped to this module's own `eager_attention_forward`, so it cannot reach
      any other architecture.
    """
    text.force_eager_attention(model)
    patch_eager_attention(attn_only_model)


class AttnOnly2LAdapter(text.DecoderTextAdapter):
    # TransformerLens prepends BOS by default, and the practice notebooks run this model through it.
    prepend_bos = True

    @classmethod
    def from_pretrained(cls, max_length: int = 128) -> "AttnOnly2LAdapter":
        hf_model, tokenizer = load_attn_only_2l()
        return cls(hf_model, tokenizer, max_length)

    def _apply_attn_lrp_patch(self) -> None:
        patch_attn_only_for_attn_lrp(self.get_model())


class TrainedAttnOnlyAdapter(AttnOnly2LAdapter):
    """Our own TinyStories models, see `attn_only_2l.train`. Subclasses name a checkpoint.

    - The whole causal LM is the model, so `lm_head` is a layer the pipeline can report on.
      - Layer paths then start at `model.`, for example `model.layers.0.self_attn.o_proj.heads.2`.
    - Training windows start at BOS, so samples do too.
    """

    default_checkpoint: ClassVar[Path]

    @classmethod
    def from_pretrained(
        cls, max_length: int = 128, checkpoint: Path | None = None
    ) -> "TrainedAttnOnlyAdapter":
        hf_model, tokenizer = load_trained_attn_only(
            checkpoint or cls.default_checkpoint
        )
        return cls(hf_model, tokenizer, max_length)

    def get_model(self) -> torch.nn.Module:
        return self.hf_model

    def embedding(self) -> torch.nn.Module:
        return self.hf_model.get_submodule("model.embed_tokens")


class TinyAttnOnly1LD128Adapter(TrainedAttnOnlyAdapter):
    default_checkpoint = Path("checkpoints/tiny_attn_only_1l_d128")
