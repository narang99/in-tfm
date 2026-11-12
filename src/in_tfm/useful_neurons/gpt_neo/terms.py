"""How each GPT-Neo neuron flows back. The rules past q, k and v are in `attention`.

Biases and the frozen norms' constants are left out of every sum, so terms add up to the
neuron's value minus a constant.
"""

from itertools import product

import torch

from ..attention import head_out_terms, o_proj_terms, score_terms
from ..flow import Terms
from ..neurons import (
    HeadOut,
    Key,
    MlpDown,
    MlpHidden,
    Neuron,
    OProj,
    PosEmbed,
    Query,
    Resid,
    ResidMid,
    Score,
    TokenEmbed,
    Value,
)
from .trace import GptNeoTrace


def terms_of(neuron: Neuron, trace: GptNeoTrace) -> Terms | None:
    match neuron:
        case TokenEmbed() | PosEmbed():
            return None
        case Resid(layer=0):
            return embedding_terms(neuron, trace)
        case Resid():
            return resid_terms(neuron, trace)
        case ResidMid():
            return resid_mid_terms(neuron, trace)
        case OProj():
            return o_proj_terms(neuron, trace.blocks[neuron.layer].attn)
        case HeadOut():
            return head_out_terms(neuron, trace.blocks[neuron.layer].attn)
        case Score():
            return score_terms(neuron, trace.blocks[neuron.layer].attn)
        case Query() | Key() | Value():
            return qkv_terms(neuron, trace)
        case MlpDown():
            return mlp_down_terms(neuron, trace)
        case MlpHidden():
            return mlp_hidden_terms(neuron, trace)
        case _:
            raise ValueError(f"no flow-back rule for {neuron}")


def embedding_terms(neuron: Resid, trace: GptNeoTrace) -> Terms:
    token = trace.token_embed[neuron.pos, neuron.dim]
    pos = trace.pos_embed[neuron.pos, neuron.dim]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        if i == 0:
            return (TokenEmbed(pos=neuron.pos, dim=neuron.dim),)
        return (PosEmbed(pos=neuron.pos, dim=neuron.dim),)

    return Terms(torch.stack([token, pos]), inputs_of)


def resid_terms(neuron: Resid, trace: GptNeoTrace) -> Terms:
    """Resid entering layer l = the mid stream of layer l-1 + that layer's MLP write."""
    below = neuron.layer - 1
    carried = trace.resid_mid[below][neuron.pos, neuron.dim]
    mlp_write = trace.blocks[below].mlp.writes[neuron.pos, neuron.dim]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        if i == 0:
            return (ResidMid(layer=below, pos=neuron.pos, dim=neuron.dim),)
        return (MlpDown(layer=below, pos=neuron.pos, dim=neuron.dim),)

    return Terms(torch.stack([carried, mlp_write]), inputs_of)


def resid_mid_terms(neuron: ResidMid, trace: GptNeoTrace) -> Terms:
    """Mid stream of layer l = resid entering layer l + each head's write."""
    carried = trace.resid[neuron.layer][neuron.pos, neuron.dim]
    head_writes = trace.blocks[neuron.layer].attn.head_writes[:, neuron.pos, neuron.dim]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        if i == 0:
            return (Resid(layer=neuron.layer, pos=neuron.pos, dim=neuron.dim),)
        return (OProj(layer=neuron.layer, pos=neuron.pos, head=i - 1, dim=neuron.dim),)

    return Terms(torch.cat([carried[None], head_writes]), inputs_of)


def qkv_terms(neuron: Query | Key | Value, trace: GptNeoTrace) -> Terms:
    block = trace.blocks[neuron.layer]
    match neuron:
        case Query():
            weights = block.attn.W_Q
        case Key():
            weights = block.attn.W_K
        case Value():
            weights = block.attn.W_V
    weight = weights[neuron.head, neuron.dim]
    normed = block.ln_1.scale[neuron.pos] * trace.resid[neuron.layer][neuron.pos]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        return (Resid(layer=neuron.layer, pos=neuron.pos, dim=i),)

    return Terms(weight * normed, inputs_of)


def mlp_down_terms(neuron: MlpDown, trace: GptNeoTrace) -> Terms:
    mlp = trace.blocks[neuron.layer].mlp
    weight = mlp.W_out[neuron.dim]
    hidden = mlp.hidden[neuron.pos]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        return (MlpHidden(layer=neuron.layer, pos=neuron.pos, dim=i),)

    return Terms(weight * hidden, inputs_of)


def mlp_hidden_terms(neuron: MlpHidden, trace: GptNeoTrace) -> Terms:
    """The activation function is per neuron, so the terms are those of its input."""
    block = trace.blocks[neuron.layer]
    weight = block.mlp.W_in[neuron.dim]
    normed = block.ln_2.scale[neuron.pos] * trace.resid_mid[neuron.layer][neuron.pos]

    def inputs_of(i: int) -> tuple[Neuron, ...]:
        return (ResidMid(layer=neuron.layer, pos=neuron.pos, dim=i),)

    return Terms(weight * normed, inputs_of)


def block_writers(trace: GptNeoTrace) -> list[Neuron]:
    head_writes: list[Neuron] = [
        OProj(layer=layer, pos=pos, head=head, dim=dim)
        for layer, block in enumerate(trace.blocks)
        for head, pos, dim in product(*map(range, block.attn.head_writes.shape))
    ]
    mlp_writes: list[Neuron] = [
        MlpDown(layer=layer, pos=pos, dim=dim)
        for layer, block in enumerate(trace.blocks)
        for pos, dim in product(*map(range, block.mlp.writes.shape))
    ]
    return head_writes + mlp_writes
