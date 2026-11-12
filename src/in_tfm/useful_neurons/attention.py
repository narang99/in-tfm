"""The attention part both models share: its trace, its keep masks, and the rules past q, k and v."""

import torch
from einops import einsum, rearrange
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
    return rearrange(
        weight, "(head head_dim) hidden -> head head_dim hidden", head=n_heads
    )


def project_per_head(
    weight: Float[torch.Tensor, "head head_dim hidden"],
    stream: Float[torch.Tensor, "seq hidden"],
) -> HeadActs:
    return einsum(
        weight, stream, "head head_dim hidden, seq hidden -> head seq head_dim"
    )


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
    scores = einsum(q, k, "head seq head_dim, head src head_dim -> head seq src")
    scores = scores * scaling
    pattern = scores.masked_fill(blocked, float("-inf")).softmax(dim=-1) * keep.pattern
    z = einsum(pattern, v, "head seq src, head src head_dim -> head seq head_dim")
    z = z * keep.z
    head_writes = einsum(
        W_O, z, "head hidden head_dim, head seq head_dim -> head seq hidden"
    )
    head_writes = head_writes * keep.head_writes
    return pattern, z, head_writes


def o_proj_terms(neuron: OProj, layer: AttnLayerTrace) -> Terms:
    weight = layer.W_O[neuron.head, neuron.dim]
    z = layer.z[neuron.head, neuron.pos]

    def inputs_of(j: int) -> tuple[Neuron, ...]:
        return (HeadOut(layer=neuron.layer, pos=neuron.pos, head=neuron.head, dim=j),)

    return Terms(weight * z, inputs_of)


def head_out_terms(neuron: HeadOut, layer: AttnLayerTrace) -> Terms:
    """z[pos, j] = sum over sources of pattern[pos, src] * v[src, j].

    A kept source keeps its value neuron only. Scores start walks of their own, see `all_scores`.
    """
    sources = slice(0, neuron.pos + 1)
    pattern = layer.pattern[neuron.head, neuron.pos, sources]
    v = layer.v[neuron.head, sources, neuron.dim]

    def inputs_of(src: int) -> tuple[Neuron, ...]:
        return (Value(layer=neuron.layer, pos=src, head=neuron.head, dim=neuron.dim),)

    return Terms(pattern * v, inputs_of)


def all_scores(layers: list[AttnLayerTrace]) -> list[Neuron]:
    """Every score whose attention weight is above 0, so masked pairs never flow back.

    Each starts its own walk and splits by q·k. This keeps the attention mix close to the
    original, so no attention weight needs zeroing.
    """
    return [
        Score(layer=layer, pos=pos, src=src, head=head)
        for layer, attn in enumerate(layers)
        for head, pos, src in (attn.pattern > 0).nonzero().tolist()
    ]


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
