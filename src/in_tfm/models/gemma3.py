from functools import partial

import torch

from .. import compat  # noqa: F401  (must precede any lxt import, see compat.py)
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import dropout_forward, gated_mlp_forward, patch_method
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

from ..attribution import patch_eager_attention
from . import text


def patch_gemma3_for_attn_lrp(model: torch.nn.Module) -> None:
    """Monkey-patch a Gemma3 model in place for AttnLRP.

    lxt ships a gemma3 patch map, and its MLP/RMSNorm rules are reused verbatim here. Only its
    attention entry is swapped: lxt's `patch_attention` is the broken-on-transformers-5.x path
    described in patch_eager_attention.

    Gemma3RMSNorm gets the same treatment LayerNorm gets in the vision path - stop-gradient
    through the normalizing statistic, so relevance flows only through the scaled input.
    """
    from lxt.efficient.models.gemma3 import gemma3_norm

    patch_map = {
        Gemma3MLP: partial(patch_method, gated_mlp_forward),
        Gemma3RMSNorm: partial(patch_method, gemma3_norm, method_name="_norm"),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        modeling_gemma3: patch_eager_attention,
    }
    model.config._attn_implementation = "eager"
    monkey_patch(model, patch_map=patch_map, verbose=True)


class Gemma3Adapter(text.DecoderTextAdapter):
    @classmethod
    def from_pretrained(
        cls, name: str = "google/gemma-3-270m", max_length: int = 128, dtype: torch.dtype = torch.float32
    ) -> "Gemma3Adapter":
        tokenizer = AutoTokenizer.from_pretrained(name)
        hf_model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        return cls(hf_model, tokenizer, max_length)

    def _apply_attn_lrp_patch(self) -> None:
        patch_gemma3_for_attn_lrp(self.get_model())
