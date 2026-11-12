"""How each neuron of the attn-only model flows back. The rules past q, k and v are in `attention`."""

from itertools import product

import torch

from ..attention import head_out_terms, o_proj_terms, score_terms
from ..flow import Terms
from ..neurons import HeadOut, Key, Neuron, OProj, PosEmbed, Query, Resid, Score, Value
from ..top_share import top_share_mask
from .trace import Trace


def terms_of(neuron: Neuron, trace: Trace) -> Terms | None:
    match neuron:
        case Resid(layer=0) | PosEmbed():
            return None
        case Resid():
            return resid_terms(neuron, trace)
        case OProj():
            return o_proj_terms(neuron, trace.layers[neuron.layer])
        case HeadOut():
            return head_out_terms(neuron, trace.layers[neuron.layer])
        case Score():
            return score_terms(neuron, trace.layers[neuron.layer])
        case Value():
            return value_terms(neuron, trace)
        case Query() | Key():
            return query_key_terms(neuron, trace)
        case _:
            raise ValueError(f"no flow-back rule for {neuron}")


def resid_terms(neuron: Resid, trace: Trace) -> Terms:
    """Resid entering layer l = resid entering layer l-1 + each head's write in layer l-1."""
    below = neuron.layer - 1
    carried = trace.resid[below][neuron.pos, neuron.dim]
    head_writes = trace.layers[below].head_writes[:, neuron.pos, neuron.dim]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        if i == 0:
            return (Resid(layer=below, pos=neuron.pos, dim=neuron.dim),)
        return (OProj(layer=below, pos=neuron.pos, head=i - 1, dim=neuron.dim),)

    return Terms(torch.cat([carried[None], head_writes]), inputs_of)


def value_terms(neuron: Value, trace: Trace) -> Terms:
    weight = trace.layers[neuron.layer].W_V[neuron.head, neuron.dim]
    resid = trace.resid[neuron.layer][neuron.pos]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        return (Resid(layer=neuron.layer, pos=neuron.pos, dim=i),)

    return Terms(weight * resid, inputs_of)


def query_key_terms(neuron: Query | Key, trace: Trace) -> Terms:
    """q and k read resid + pos, so each input dim gives one resid term and one position term."""
    layer = trace.layers[neuron.layer]
    weights = layer.W_Q if isinstance(neuron, Query) else layer.W_K
    weight = weights[neuron.head, neuron.dim]
    resid = trace.resid[neuron.layer][neuron.pos]
    pos_embed = trace.pos_embed[neuron.pos]
    hidden = resid.shape[0]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        if i < hidden:
            return (Resid(layer=neuron.layer, pos=neuron.pos, dim=i),)
        return (PosEmbed(pos=neuron.pos, dim=i - hidden),)

    return Terms(torch.cat([weight * resid, weight * pos_embed]), inputs_of)


def block_writers(trace: Trace) -> list[Neuron]:
    return [
        OProj(layer=layer, pos=pos, head=head, dim=dim)
        for layer, attn in enumerate(trace.layers)
        for head, pos, dim in product(*map(range, attn.head_writes.shape))
    ]


def top_share_writers(trace: Trace, share: float) -> list[Neuron]:
    """Each head's write vector at each token keeps its top-share dims, per sign."""
    return [
        OProj(layer=layer, pos=pos, head=head, dim=dim)
        for layer, attn in enumerate(trace.layers)
        for head, pos, dim in top_share_mask(attn.head_writes, share).nonzero().tolist()
    ]
