"""Which neurons a traced GPT-Neo forward pass keeps. Every other neuron is zeroed."""

import torch
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict
from transformers import GPTNeoConfig

from ..attention import AttnLayerKeep, attn_layer_keep, mark_attention_kept
from ..neurons import (
    MlpDown,
    MlpHidden,
    Neuron,
    PosEmbed,
    Resid,
    ResidMid,
    TokenEmbed,
)

StreamMask = Bool[torch.Tensor, "seq hidden"]


class GptNeoKeep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_embed: StreamMask
    pos_embed: StreamMask
    resid: list[StreamMask]
    resid_mid: list[StreamMask]
    layers: list[AttnLayerKeep]
    mlp_hidden: list[Bool[torch.Tensor, "seq mlp"]]
    mlp_down: list[StreamMask]


def mlp_size(config: GPTNeoConfig) -> int:
    return config.intermediate_size or 4 * config.hidden_size


def keep_all(config: GPTNeoConfig, seq: int, fill: bool = True) -> GptNeoKeep:
    hidden, n_layers = config.hidden_size, config.num_layers
    heads = config.num_heads
    head_dim = hidden // heads

    def stream() -> torch.Tensor:
        return torch.full((seq, hidden), fill)

    return GptNeoKeep(
        token_embed=stream(),
        pos_embed=stream(),
        resid=[stream() for _ in range(n_layers + 1)],
        resid_mid=[stream() for _ in range(n_layers)],
        layers=[
            attn_layer_keep(heads, seq, head_dim, hidden, fill) for _ in range(n_layers)
        ],
        mlp_hidden=[torch.full((seq, mlp_size(config)), fill) for _ in range(n_layers)],
        mlp_down=[stream() for _ in range(n_layers)],
    )


def keep_only(neurons: list[Neuron], config: GPTNeoConfig, seq: int) -> GptNeoKeep:
    keep = keep_all(config, seq, fill=False)
    for neuron in neurons:
        mark_kept(keep, neuron)
    return keep


def mark_kept(keep: GptNeoKeep, neuron: Neuron) -> None:
    match neuron:
        case TokenEmbed(pos=pos, dim=dim):
            keep.token_embed[pos, dim] = True
        case PosEmbed(pos=pos, dim=dim):
            keep.pos_embed[pos, dim] = True
        case Resid(layer=layer, pos=pos, dim=dim):
            keep.resid[layer][pos, dim] = True
        case ResidMid(layer=layer, pos=pos, dim=dim):
            keep.resid_mid[layer][pos, dim] = True
        case MlpHidden(layer=layer, pos=pos, dim=dim):
            keep.mlp_hidden[layer][pos, dim] = True
        case MlpDown(layer=layer, pos=pos, dim=dim):
            keep.mlp_down[layer][pos, dim] = True
        case _:
            mark_attention_kept(keep.layers, neuron)


def with_stream(keep: GptNeoKeep) -> GptNeoKeep:
    keep.token_embed.fill_(True)
    keep.pos_embed.fill_(True)
    for stream in keep.resid + keep.resid_mid:
        stream.fill_(True)
    return keep


def with_stream_and_writers(keep: GptNeoKeep) -> GptNeoKeep:
    """Block mode keeps the residual stream and every writer, and trims only inside blocks."""
    for mlp_down in with_stream(keep).mlp_down:
        mlp_down.fill_(True)
    for layer in keep.layers:
        layer.head_writes.fill_(True)
    return keep
