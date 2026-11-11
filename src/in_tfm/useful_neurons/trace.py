"""Every activation and weight the walk reads, for one sequence.

Recomputed from the weights rather than hooked out of the forward pass, so each tensor has the
exact layout the term rules index into. `test_useful_neurons` checks it reproduces the model's logits.
"""

from typing import cast

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from ..models.attn_only_2l.model import AttnOnlyAttention, AttnOnlyForCausalLM

HeadActs = Float[torch.Tensor, "head seq head_dim"]


class AttnLayerTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    W_Q: Float[torch.Tensor, "head head_dim hidden"]
    W_K: Float[torch.Tensor, "head head_dim hidden"]
    W_V: Float[torch.Tensor, "head head_dim hidden"]
    W_O: Float[torch.Tensor, "head hidden head_dim"]
    scaling: float

    q: HeadActs
    k: HeadActs
    v: HeadActs
    pattern: Float[torch.Tensor, "head seq src"]
    z: HeadActs
    head_writes: Float[torch.Tensor, "head seq hidden"]


class Trace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, " seq"]
    W_U: Float[torch.Tensor, "vocab hidden"]
    pos_embed: Float[torch.Tensor, "seq hidden"]
    resid: list[Float[torch.Tensor, "seq hidden"]]
    """`resid[l]` is the stream entering layer l, so `resid[-1]` is what the unembed reads."""
    layers: list[AttnLayerTrace]

    @property
    def final_layer(self) -> int:
        return len(self.layers)


def per_head(
    weight: Float[torch.Tensor, "inner hidden"], n_heads: int
) -> Float[torch.Tensor, "head head_dim hidden"]:
    return weight.view(n_heads, -1, weight.shape[-1])


def trace_attn_layer(
    attn: AttnOnlyAttention,
    resid: Float[torch.Tensor, "seq hidden"],
    pos_embed: Float[torch.Tensor, "seq hidden"],
) -> AttnLayerTrace:
    """Biases are skipped. They belong to no neuron, and the trained 1L models have none."""
    n_heads = attn.n_heads
    W_Q = per_head(attn.q_proj.weight, n_heads)
    W_K = per_head(attn.k_proj.weight, n_heads)
    W_V = per_head(attn.v_proj.weight, n_heads)
    W_O = torch.stack([attn.o_proj.head_weight(h) for h in range(n_heads)])

    qk_input = resid + pos_embed
    q = torch.einsum("hjd,sd->hsj", W_Q, qk_input)
    k = torch.einsum("hjd,sd->hsj", W_K, qk_input)
    v = torch.einsum("hjd,sd->hsj", W_V, resid)

    scores = torch.einsum("hpj,hsj->hps", q, k) * attn.scaling
    seq = resid.shape[0]
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    pattern = scores.masked_fill(future, float("-inf")).softmax(dim=-1)

    z = torch.einsum("hps,hsj->hpj", pattern, v)
    head_writes = torch.einsum("hdj,hpj->hpd", W_O, z)
    return AttnLayerTrace(
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        scaling=attn.scaling,
        q=q,
        k=k,
        v=v,
        pattern=pattern,
        z=z,
        head_writes=head_writes,
    )


@torch.no_grad()
def trace_sequence(
    model: AttnOnlyForCausalLM, tokens: Int[torch.Tensor, " seq"]
) -> Trace:
    stack = model.model
    pos_embed = stack.pos_embed.weight[: tokens.shape[0]]
    resid = [stack.embed_tokens.weight[tokens]]
    layers = []
    for layer in stack.layers:
        attn = cast(AttnOnlyAttention, layer.self_attn)
        layer_trace = trace_attn_layer(attn, resid[-1], pos_embed)
        layers.append(layer_trace)
        resid.append(resid[-1] + layer_trace.head_writes.sum(dim=0))
    return Trace(
        tokens=tokens,
        W_U=model.lm_head.weight,
        pos_embed=pos_embed,
        resid=resid,
        layers=layers,
    )
