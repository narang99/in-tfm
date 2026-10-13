"""AttnLRP on a head's OV output, against its exact per-token contribution.

- The neuron is coordinate `h * d_head + d` of the attention-weighted values, read at the `=` position.
  - It equals `sum over source tokens of attention(=, source) * value(source)[h, d]`.
  - So the exact share each of `a`, `b` and `=` contributes is known, and AttnLRP can be checked against it.
- A value neuron is not a test of this, see `GrokkingAdapter`: its relevance sits on its own token whatever the method.
"""

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel

from .attribution import compute_attnlrp_relevance
from .models.grokking import GrokTransformer
from .models.grokking.source import EMBEDS_KEY
from .sources import ModelBatch

EQUALS_POSITION = 2
Tokens = Int[torch.Tensor, "n 3"]
PerToken = Float[np.ndarray, "n 3"]


def z_getter(layer_idx: int):
    def getter(model: torch.nn.Module) -> torch.nn.Module:
        return model.layers[layer_idx].self_attn.mixed

    return getter


def batch_of(model: GrokTransformer, tokens: Tokens) -> ModelBatch:
    with torch.no_grad():
        embeds = model.embed_tokens(tokens)
    return ModelBatch(
        kwargs={EMBEDS_KEY: embeds}, grad_leaf_key=EMBEDS_KEY, valid_mask=torch.ones_like(tokens, dtype=torch.bool), display_ids=tokens
    )


@torch.no_grad()
def exact_contributions(model: GrokTransformer, tokens: Tokens, neuron: int) -> tuple[PerToken, Float[np.ndarray, "n"]]:
    """Per source token `attention * value` at the `=` position, and the neuron's value there."""
    attn = model.layers[0].self_attn
    hidden = model.embed_tokens(tokens) + model.pos_embed
    heads = lambda proj: attn._to_heads(proj(hidden))
    query, key, value = heads(attn.q_proj), heads(attn.k_proj), heads(attn.v_proj)
    head, dim = divmod(neuron, attn.d_head)
    weights = torch.softmax(
        (query[:, head, EQUALS_POSITION, None, :] * key[:, head]).sum(-1) * attn.scaling, dim=-1
    )
    contributions = weights * value[:, head, :, dim]
    return contributions.numpy(), contributions.sum(-1).numpy()


def attnlrp_per_token(model: GrokTransformer, tokens: Tokens, neuron: int) -> PerToken:
    """Summed over the embedding, like the text path. Needs `patch_for_attn_lrp` to have been applied for AttnLRP."""
    relevance = compute_attnlrp_relevance(model, batch_of(model, tokens), z_getter(0), neuron, EQUALS_POSITION)
    return relevance.sum(-1)


class Agreement(BaseModel):
    neuron: int
    pearson: float
    """Between relevance and the exact contribution, over every (sample, source token)."""
    conserved: float
    """Least-squares slope of the relevance summed over tokens on the neuron's value, over samples.

    - A slope is used because a ratio per sample explodes where the value is near zero.
    - The leaf is the token embedding, so the share that flows to the position embedding is not counted.
    - So a complete method still lands below 1 here, by that share.
    """
    share_on_largest_contributor: float
    """How often the token with the most relevance is the one with the largest exact contribution."""


def agreement(relevance: PerToken, contributions: PerToken, value: Float[np.ndarray, "n"], neuron: int) -> Agreement:
    return Agreement(
        neuron=neuron,
        pearson=float(np.corrcoef(relevance.ravel(), contributions.ravel())[0, 1]),
        conserved=float(relevance.sum(-1) @ value / (value @ value)),
        share_on_largest_contributor=float(np.mean(relevance.argmax(-1) == contributions.argmax(-1))),
    )
