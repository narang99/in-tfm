from pathlib import Path

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


class TinyAttnOnlyAdapter(AttnOnly2LAdapter):
    """Our own trained checkpoints, see `attn_only_2l.train`.

    - `get_model()` is the whole causal LM, so `lm_head` is a layer, and a logit is a neuron.
      - Its decoder layers are then `model.layers.<i>`, not `layers.<i>`.
    """

    embed_path = "model.embed_tokens"

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: Path, max_length: int = 128
    ) -> "TinyAttnOnlyAdapter":
        hf_model, tokenizer = load_trained_attn_only(checkpoint_dir)
        return cls(hf_model, tokenizer, max_length)

    def get_model(self) -> torch.nn.Module:
        return self.hf_model
