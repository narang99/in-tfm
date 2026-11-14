"""Which writes of one layer the stream-writers walk keeps, at every token of one sequence.

- A write of layer L only enters the walk through a stream dim read after it.
  - Layer L's MLP reads its head writes. Later layers read both kinds.
  - So the walk stops at layer L's attention and everything below, and the picks of layer L do not change.
- Scores start no walks. The QK circuit is studied separately.
"""

from functools import partial

import torch
from jaxtyping import Bool

from ..flow import Terms, TermsOf
from ..neurons import (
    MlpDown,
    MlpHidden,
    Neuron,
    OProj,
    PosEmbed,
    Resid,
    ResidMid,
    TokenEmbed,
)
from ..walk import AllLogits, find_useful_neurons
from .terms import terms_with_stream_writers
from .trace import GptNeoTrace


def walked_above(layer: int, terms_of: TermsOf) -> TermsOf:
    def terms_above(neuron: Neuron) -> Terms | None:
        match neuron:
            case TokenEmbed() | PosEmbed():
                return None
            case Resid() | ResidMid():
                return terms_of(neuron)
            case MlpDown() | MlpHidden() if neuron.layer >= layer:
                return terms_of(neuron)
            case _ if neuron.layer > layer:
                return terms_of(neuron)
        return None

    return terms_above


def picked_writers(trace: GptNeoTrace, layer: int, share: float) -> list[Neuron]:
    terms_of = partial(terms_with_stream_writers, trace=trace)
    useful = find_useful_neurons(
        trace.unembed, walked_above(layer, terms_of), AllLogits(), share, scores=[]
    )
    return [
        n for n in useful.picks if isinstance(n, (MlpDown, OProj)) and n.layer == layer
    ]


def mlp_write_picks(
    trace: GptNeoTrace, layer: int, share: float
) -> Bool[torch.Tensor, "seq hidden"]:
    picked = torch.zeros(trace.resid[0].shape, dtype=torch.bool)
    for n in picked_writers(trace, layer, share):
        if isinstance(n, MlpDown):
            picked[n.pos, n.dim] = True
    return picked


def head_write_picks(
    trace: GptNeoTrace, layer: int, share: float
) -> Bool[torch.Tensor, "head seq hidden"]:
    picked = torch.zeros(trace.blocks[layer].attn.head_writes.shape, dtype=torch.bool)
    for n in picked_writers(trace, layer, share):
        if isinstance(n, OProj):
            picked[n.head, n.pos, n.dim] = True
    return picked
