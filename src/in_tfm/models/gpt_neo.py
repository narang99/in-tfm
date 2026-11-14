"""GPT-Neo decoders, such as TinyStories-1M.

- The AttnLRP rules follow lxt's GPT-2 map, since lxt has no GPT-Neo one.
  - GPT-Neo computes attention in `GPTNeoSelfAttention._attn`, not in an `eager_attention_forward`.
    So the uniform rule is applied there, on q, k and v, as lxt's `wrap_attention_forward` does.
- The patches act on the classes, so they reach every GPT-Neo model in the process.
"""

from collections.abc import Sequence
from functools import partial

import torch

# isort: split
from .. import compat  # noqa: F401  (must precede any lxt import, see compat.py)

# isort: split
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import dropout_forward, layer_norm_forward, patch_method
from lxt.efficient.rules import divide_gradient, identity_rule_implicit
from transformers import GPTNeoForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP, GPTNeoSelfAttention

from ..sources import TextSource
from . import text

ORIGINAL_ATTN = GPTNeoSelfAttention._attn


def mlp_forward(self: GPTNeoMLP, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.c_fc(hidden_states)
    hidden_states = identity_rule_implicit(self.act, hidden_states)
    return self.c_proj(hidden_states)


def attn_with_uniform_rule(
    self: GPTNeoSelfAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    """q and k each get a quarter of the relevance through the q.k and softmax products, v a half."""
    return ORIGINAL_ATTN(
        self,
        divide_gradient(query, 4),
        divide_gradient(key, 4),
        divide_gradient(value, 2),
        attention_mask,
    )


def patch_gpt_neo_for_attn_lrp(model: torch.nn.Module) -> None:
    patch_map = {
        GPTNeoMLP: partial(patch_method, mlp_forward),
        torch.nn.LayerNorm: partial(patch_method, layer_norm_forward),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        GPTNeoSelfAttention: partial(
            patch_method, attn_with_uniform_rule, method_name="_attn"
        ),
    }
    text.force_eager_attention(model)
    monkey_patch(model, patch_map=patch_map, verbose=True)


class GptNeoAdapter(text.DecoderTextAdapter):
    """Holds the bare `GPTNeoModel`. Layer names start below `transformer`, such as `h.6.mlp.c_proj`."""

    def get_model(self) -> torch.nn.Module:
        return self.hf_model.get_submodule("transformer")

    @classmethod
    def from_pretrained(cls, name: str, max_length: int = 128) -> "GptNeoAdapter":
        model = GPTNeoForCausalLM.from_pretrained(name).eval()
        model.requires_grad_(False)
        return cls(model, text.load_tokenizer(name), max_length)

    def make_source(self, samples: Sequence[str]) -> TextSource:
        return TextSource(
            samples,
            self.tokenizer,
            self.get_model().get_submodule("wte"),
            self.max_length,
            self.prepend_bos,
            self.bos_is_a_hit,
        )

    def _apply_attn_lrp_patch(self) -> None:
        patch_gpt_neo_for_attn_lrp(self.get_model())
