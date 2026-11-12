"""The attention part both models share: its trace, its keep masks, and the rules past q, k and v."""

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict

from .flow import Terms
from .neurons import HeadOut, Key, Neuron, OProj, Query, Score, Value

HeadActs = Float[torch.Tensor, "head seq head_dim"]
HeadMask = Bool[torch.Tensor, "head seq head_dim"]


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


class AttnLayerKeep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    q: HeadMask
    k: HeadMask
    v: HeadMask
    pattern: Bool[torch.Tensor, "head seq src"]
    """A dropped score zeroes its attention weight, which removes exactly its term from z."""
    z: HeadMask
    head_writes: Bool[torch.Tensor, "head seq hidden"]


def attn_layer_keep(
    heads: int, seq: int, head_dim: int, hidden: int, fill: bool
) -> AttnLayerKeep:
    def mask(*shape: int) -> torch.Tensor:
        return torch.full(shape, fill, dtype=torch.bool)

    return AttnLayerKeep(
        q=mask(heads, seq, head_dim),
        k=mask(heads, seq, head_dim),
        v=mask(heads, seq, head_dim),
        pattern=mask(heads, seq, seq),
        z=mask(heads, seq, head_dim),
        head_writes=mask(heads, seq, hidden),
    )


def per_head(
    weight: Float[torch.Tensor, "inner hidden"], n_heads: int
) -> Float[torch.Tensor, "head head_dim hidden"]:
    return weight.view(n_heads, -1, weight.shape[-1])


def attend(
    q: HeadActs,
    k: HeadActs,
    v: HeadActs,
    W_O: Float[torch.Tensor, "head hidden head_dim"],
    scaling: float,
    blocked: Bool[torch.Tensor, "seq src"],
    keep: AttnLayerKeep,
) -> tuple[
    Float[torch.Tensor, "head seq src"],
    HeadActs,
    Float[torch.Tensor, "head seq hidden"],
]:
    scores = torch.einsum("hpj,hsj->hps", q, k) * scaling
    pattern = scores.masked_fill(blocked, float("-inf")).softmax(dim=-1) * keep.pattern
    z = torch.einsum("hps,hsj->hpj", pattern, v) * keep.z
    head_writes = torch.einsum("hdj,hpj->hpd", W_O, z) * keep.head_writes
    return pattern, z, head_writes


def o_proj_terms(neuron: OProj, layer: AttnLayerTrace) -> Terms:
    weight = layer.W_O[neuron.head, neuron.dim]
    z = layer.z[neuron.head, neuron.pos]

    def inputs_of(j: int) -> tuple[Neuron, ...]:
        return (HeadOut(layer=neuron.layer, pos=neuron.pos, head=neuron.head, dim=j),)

    return Terms(weight * z, inputs_of)


def head_out_terms(neuron: HeadOut, layer: AttnLayerTrace) -> Terms:
    """z[pos, j] = sum over sources of pattern[pos, src] * v[src, j].

    Keeping a source keeps both its value neuron and the score that weighted it.
    """
    sources = slice(0, neuron.pos + 1)
    pattern = layer.pattern[neuron.head, neuron.pos, sources]
    v = layer.v[neuron.head, sources, neuron.dim]

    def inputs_of(src: int) -> tuple[Neuron, ...]:
        return (
            Value(layer=neuron.layer, pos=src, head=neuron.head, dim=neuron.dim),
            Score(layer=neuron.layer, pos=neuron.pos, src=src, head=neuron.head),
        )

    return Terms(pattern * v, inputs_of)


def score_terms(neuron: Score, layer: AttnLayerTrace) -> Terms:
    """score = scaling * sum over j of q[pos, j] * k[src, j]. A kept j keeps both q and k."""
    q = layer.q[neuron.head, neuron.pos]
    k = layer.k[neuron.head, neuron.src]

    def inputs_of(j: int) -> tuple[Neuron, ...]:
        return (
            Query(layer=neuron.layer, pos=neuron.pos, head=neuron.head, dim=j),
            Key(layer=neuron.layer, pos=neuron.src, head=neuron.head, dim=j),
        )

    return Terms(layer.scaling * q * k, inputs_of)


def mark_attention_kept(layers: list[AttnLayerKeep], neuron: Neuron) -> None:
    match neuron:
        case OProj(layer=layer, pos=pos, head=head, dim=dim):
            layers[layer].head_writes[head, pos, dim] = True
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            layers[layer].z[head, pos, dim] = True
        case Value(layer=layer, pos=pos, head=head, dim=dim):
            layers[layer].v[head, pos, dim] = True
        case Query(layer=layer, pos=pos, head=head, dim=dim):
            layers[layer].q[head, pos, dim] = True
        case Key(layer=layer, pos=pos, head=head, dim=dim):
            layers[layer].k[head, pos, dim] = True
        case Score(layer=layer, pos=pos, src=src, head=head):
            layers[layer].pattern[head, pos, src] = True
