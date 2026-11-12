"""Which neurons a traced forward pass keeps. Every other neuron is zeroed."""

import torch
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict

from ..models.attn_only_2l.model import AttnOnlyConfig
from .neurons import (
    HeadOut,
    Key,
    Logit,
    Neuron,
    OProj,
    PosEmbed,
    Query,
    Resid,
    Score,
    Value,
)

HeadMask = Bool[torch.Tensor, "head seq head_dim"]


class AttnLayerKeep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    q: HeadMask
    k: HeadMask
    v: HeadMask
    pattern: Bool[torch.Tensor, "head seq src"]
    """A dropped score zeroes its attention weight, which removes exactly its term from z."""
    z: HeadMask
    head_writes: Bool[torch.Tensor, "head seq hidden"]


class Keep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resid: list[Bool[torch.Tensor, "seq hidden"]]
    pos_embed: Bool[torch.Tensor, "seq hidden"]
    layers: list[AttnLayerKeep]


def keep_all(config: AttnOnlyConfig, seq: int, fill: bool = True) -> Keep:
    heads, head_dim, hidden = (
        config.num_attention_heads,
        config.head_dim,
        config.hidden_size,
    )

    def mask(*shape: int) -> torch.Tensor:
        return torch.full(shape, fill, dtype=torch.bool)

    def layer_keep() -> AttnLayerKeep:
        return AttnLayerKeep(
            q=mask(heads, seq, head_dim),
            k=mask(heads, seq, head_dim),
            v=mask(heads, seq, head_dim),
            pattern=mask(heads, seq, seq),
            z=mask(heads, seq, head_dim),
            head_writes=mask(heads, seq, hidden),
        )

    n_layers = config.num_hidden_layers
    return Keep(
        resid=[mask(seq, hidden) for _ in range(n_layers + 1)],
        pos_embed=mask(seq, hidden),
        layers=[layer_keep() for _ in range(n_layers)],
    )


def keep_only(neurons: list[Neuron], config: AttnOnlyConfig, seq: int) -> Keep:
    keep = keep_all(config, seq, fill=False)
    for neuron in neurons:
        mark_kept(keep, neuron)
    return keep


def mark_kept(keep: Keep, neuron: Neuron) -> None:
    match neuron:
        case Resid(layer=layer, pos=pos, dim=dim):
            keep.resid[layer][pos, dim] = True
        case PosEmbed(pos=pos, dim=dim):
            keep.pos_embed[pos, dim] = True
        case OProj(layer=layer, pos=pos, head=head, dim=dim):
            keep.layers[layer].head_writes[head, pos, dim] = True
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            keep.layers[layer].z[head, pos, dim] = True
        case Value(layer=layer, pos=pos, head=head, dim=dim):
            keep.layers[layer].v[head, pos, dim] = True
        case Query(layer=layer, pos=pos, head=head, dim=dim):
            keep.layers[layer].q[head, pos, dim] = True
        case Key(layer=layer, pos=pos, head=head, dim=dim):
            keep.layers[layer].k[head, pos, dim] = True
        case Score(layer=layer, pos=pos, src=src, head=head):
            keep.layers[layer].pattern[head, pos, src] = True
        case Logit():
            pass
