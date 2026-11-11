"""How each neuron flows back: its value as a sum of terms, and the neurons behind each term."""

from collections.abc import Callable
from typing import NamedTuple

import torch
from jaxtyping import Float

from .neurons import HeadOut, Key, Neuron, OProj, PosEmbed, Query, Resid, Score, Value
from .trace import Trace


class Terms(NamedTuple):
    values: Float[torch.Tensor, " n"]
    inputs_of: Callable[[int], tuple[Neuron, ...]]
    """Built lazily, so only the kept terms ever create neurons."""


def terms_of(neuron: Neuron, trace: Trace) -> Terms | None:
    match neuron:
        case Resid(layer=0) | PosEmbed():
            return None
        case Resid():
            return resid_terms(neuron, trace)
        case OProj():
            return o_proj_terms(neuron, trace)
        case HeadOut():
            return head_out_terms(neuron, trace)
        case Value():
            return value_terms(neuron, trace)
        case Score():
            return score_terms(neuron, trace)
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


def o_proj_terms(neuron: OProj, trace: Trace) -> Terms:
    layer = trace.layers[neuron.layer]
    weight = layer.W_O[neuron.head, neuron.dim]
    z = layer.z[neuron.head, neuron.pos]

    def inputs_of(j: int) -> tuple[Neuron, ...]:
        return (HeadOut(layer=neuron.layer, pos=neuron.pos, head=neuron.head, dim=j),)

    return Terms(weight * z, inputs_of)


def head_out_terms(neuron: HeadOut, trace: Trace) -> Terms:
    """z[pos, j] = sum over sources of pattern[pos, src] * v[src, j].

    Keeping a source keeps both its value neuron and the score that weighted it.
    """
    layer = trace.layers[neuron.layer]
    sources = slice(0, neuron.pos + 1)
    pattern = layer.pattern[neuron.head, neuron.pos, sources]
    v = layer.v[neuron.head, sources, neuron.dim]

    def inputs_of(src: int) -> tuple[Neuron, ...]:
        return (
            Value(layer=neuron.layer, pos=src, head=neuron.head, dim=neuron.dim),
            Score(layer=neuron.layer, pos=neuron.pos, src=src, head=neuron.head),
        )

    return Terms(pattern * v, inputs_of)


def value_terms(neuron: Value, trace: Trace) -> Terms:
    weight = trace.layers[neuron.layer].W_V[neuron.head, neuron.dim]
    resid = trace.resid[neuron.layer][neuron.pos]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        return (Resid(layer=neuron.layer, pos=neuron.pos, dim=i),)

    return Terms(weight * resid, inputs_of)


def score_terms(neuron: Score, trace: Trace) -> Terms:
    """score = scaling * sum over j of q[pos, j] * k[src, j]. A kept j keeps both q and k."""
    layer = trace.layers[neuron.layer]
    q = layer.q[neuron.head, neuron.pos]
    k = layer.k[neuron.head, neuron.src]

    def inputs_of(j: int) -> tuple[Neuron, ...]:
        return (
            Query(layer=neuron.layer, pos=neuron.pos, head=neuron.head, dim=j),
            Key(layer=neuron.layer, pos=neuron.src, head=neuron.head, dim=j),
        )

    return Terms(layer.scaling * q * k, inputs_of)


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
