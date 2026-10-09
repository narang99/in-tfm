import torch

from ...attribution import patch_eager_attention
from .. import text
from . import model as attn_only_model
from .model import load_attn_only_2l


def patch_attn_only_for_attn_lrp(model: torch.nn.Module) -> None:
    """Attention is the only rule needed.

    - The model has no normalization and no MLP, so softmax is its only nonlinearity.
    - Everything else is linear in its input, where plain gradient*input is a valid LRP rule.
    - The patch is scoped to this module's own `eager_attention_forward`, so it cannot reach
      any other architecture.
    """
    model.config._attn_implementation = "eager"
    patch_eager_attention(attn_only_model)


class AttnOnly2LAdapter(text.DecoderTextAdapter):
    @classmethod
    def from_pretrained(cls, max_length: int = 128) -> "AttnOnly2LAdapter":
        hf_model, tokenizer = load_attn_only_2l()
        return cls(hf_model, tokenizer, max_length)

    def _apply_attn_lrp_patch(self) -> None:
        patch_attn_only_for_attn_lrp(self.get_model())
