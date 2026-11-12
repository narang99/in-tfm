"""Every activation and weight the walk reads, for one sequence.

Recomputed from the weights rather than hooked out of the forward pass, so each tensor has the
exact layout the term rules index into. `test_useful_neurons` checks it reproduces the model's logits.
"""

from typing import cast

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from ...models.attn_only_2l.model import AttnOnlyAttention, AttnOnlyForCausalLM
from ..attention import AttnLayerKeep, AttnLayerTrace, attend, per_head
from ..flow import Unembed
from ..neuron_means import SiteActivations
from ..replacement import LayerFill, Replace, Site, at_layer, keep_or_fill, zeros
from .keep import Keep, keep_all


class Trace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, " seq"]
    W_U: Float[torch.Tensor, "vocab hidden"]
    b_U: Float[torch.Tensor, " vocab"]
    pos_embed: Float[torch.Tensor, "seq hidden"]
    resid: list[Float[torch.Tensor, "seq hidden"]]
    """`resid[l]` is the stream entering layer l, so `resid[-1]` is what the unembed reads."""
    layers: list[AttnLayerTrace]

    @property
    def unembed(self) -> Unembed:
        return Unembed(
            W_U=self.W_U, inputs=self.resid[-1], final_layer=len(self.layers)
        )

    @property
    def logits(self) -> Float[torch.Tensor, "seq vocab"]:
        return self.resid[-1] @ self.W_U.T + self.b_U


def trace_attn_layer(
    attn: AttnOnlyAttention,
    resid: Float[torch.Tensor, "seq hidden"],
    pos_embed: Float[torch.Tensor, "seq hidden"],
    keep: AttnLayerKeep,
    fill: LayerFill,
) -> AttnLayerTrace:
    """Biases are skipped. They belong to no neuron, and the trained 1L models have none."""
    n_heads = attn.n_heads
    W_Q = per_head(attn.q_proj.weight, n_heads)
    W_K = per_head(attn.k_proj.weight, n_heads)
    W_V = per_head(attn.v_proj.weight, n_heads)
    W_O = torch.stack([attn.o_proj.head_weight(h) for h in range(n_heads)])

    qk_input = resid + pos_embed
    q = keep_or_fill(torch.einsum("hjd,sd->hsj", W_Q, qk_input), keep.q, fill("q"))
    k = keep_or_fill(torch.einsum("hjd,sd->hsj", W_K, qk_input), keep.k, fill("k"))
    v = keep_or_fill(torch.einsum("hjd,sd->hsj", W_V, resid), keep.v, fill("v"))

    seq = resid.shape[0]
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    pattern, z, head_writes = attend(q, k, v, W_O, attn.scaling, future, keep, fill)
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
    model: AttnOnlyForCausalLM,
    tokens: Int[torch.Tensor, " seq"],
    keep: Keep | None = None,
    replace: Replace = zeros,
) -> Trace:
    seq = tokens.shape[0]
    keep = keep or keep_all(model.config, seq)
    stack = model.model
    pos_embed = keep_or_fill(
        stack.pos_embed.weight[:seq], keep.pos_embed, replace(Site(name="pos_embed"))
    )
    embedded = stack.embed_tokens.weight[tokens]
    resid = [
        keep_or_fill(embedded, keep.resid[0], replace(Site(name="resid", layer=0)))
    ]
    layers = []
    for idx, layer in enumerate(stack.layers):
        attn = cast(AttnOnlyAttention, layer.self_attn)
        layer_trace = trace_attn_layer(
            attn, resid[-1], pos_embed, keep.layers[idx], at_layer(replace, idx)
        )
        layers.append(layer_trace)
        written = resid[-1] + layer_trace.head_writes.sum(dim=0)
        fill = replace(Site(name="resid", layer=idx + 1))
        resid.append(keep_or_fill(written, keep.resid[idx + 1], fill))
    return Trace(
        tokens=tokens,
        W_U=model.lm_head.weight,
        b_U=model.lm_head.bias,
        pos_embed=pos_embed,
        resid=resid,
        layers=layers,
    )


def site_activations(trace: Trace) -> SiteActivations:
    """Every site the ablation can fill, with its sequence axis. Attention weights are left out."""
    sites: SiteActivations = {Site(name="pos_embed"): (trace.pos_embed, 0)}
    for layer, resid in enumerate(trace.resid):
        sites[Site(name="resid", layer=layer)] = (resid, 0)
    for layer, attn in enumerate(trace.layers):
        for name in ("q", "k", "v", "z", "head_writes"):
            sites[Site(name=name, layer=layer)] = (getattr(attn, name), 1)
    return sites
