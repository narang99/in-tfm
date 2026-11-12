"""Which neurons a traced forward pass keeps. Every other neuron is zeroed."""

import torch
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict

from ...models.attn_only_2l.model import AttnOnlyConfig
from ..attention import AttnLayerKeep, attn_layer_keep, mark_attention_kept
from ..neurons import Neuron, PosEmbed, Resid


class Keep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resid: list[Bool[torch.Tensor, "seq hidden"]]
    pos_embed: Bool[torch.Tensor, "seq hidden"]
    layers: list[AttnLayerKeep]


def keep_all(config: AttnOnlyConfig, seq: int, fill: bool = True) -> Keep:
    hidden, n_layers = config.hidden_size, config.num_hidden_layers
    heads, head_dim = config.num_attention_heads, config.head_dim
    return Keep(
        resid=[torch.full((seq, hidden), fill) for _ in range(n_layers + 1)],
        pos_embed=torch.full((seq, hidden), fill),
        layers=[
            attn_layer_keep(heads, seq, head_dim, hidden, fill) for _ in range(n_layers)
        ],
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
        case _:
            mark_attention_kept(keep.layers, neuron)


def with_stream(keep: Keep) -> Keep:
    for resid in keep.resid:
        resid.fill_(True)
    keep.pos_embed.fill_(True)
    return keep


def with_stream_and_writers(keep: Keep) -> Keep:
    """Block mode keeps the residual stream and every head's write, and trims only inside blocks."""
    for layer in with_stream(keep).layers:
        layer.head_writes.fill_(True)
    return keep
