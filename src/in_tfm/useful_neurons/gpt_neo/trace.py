"""Every activation and weight the walk reads from GPT-Neo, for one sequence.

LayerNorm is kept linear by freezing each token's mean and std.
- A neuron reading through a norm sees `scale * x + constant` per dim, with `scale = weight / std`.
- So input dim d's term is `W[j, d] * scale[d] * x[d]`. The mean and the norm's bias are constants, like biases.
- The ablated pass reuses the original pass's stats, so zeroing x[d] removes exactly that term.
"""

from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from torch import nn
from transformers import GPTNeoForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoBlock,
    GPTNeoSelfAttention,
)

from ..attention import AttnLayerKeep, AttnLayerTrace, attend, per_head
from ..flow import Unembed
from .keep import GptNeoKeep, keep_all

Stream = Float[torch.Tensor, "seq hidden"]


class NormStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: Float[torch.Tensor, "seq 1"]
    std: Float[torch.Tensor, "seq 1"]


class LayerNormTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stats: NormStats
    scale: Stream
    output: Stream


class MlpTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    W_in: Float[torch.Tensor, "mlp hidden"]
    W_out: Float[torch.Tensor, "hidden mlp"]
    hidden: Float[torch.Tensor, "seq mlp"]
    writes: Stream
    """`W_out @ hidden`, without the output bias."""


class BlockTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ln_1: LayerNormTrace
    attn: AttnLayerTrace
    ln_2: LayerNormTrace
    mlp: MlpTrace


class FrozenNorms(BaseModel):
    ln_1: list[NormStats]
    ln_2: list[NormStats]
    ln_f: NormStats


class GptNeoTrace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Int[torch.Tensor, " seq"]
    W_U: Float[torch.Tensor, "vocab hidden"]
    token_embed: Stream
    pos_embed: Stream
    resid: list[Stream]
    """`resid[l]` enters block l, so `resid[-1]` is what the final norm reads."""
    resid_mid: list[Stream]
    blocks: list[BlockTrace]
    ln_f: LayerNormTrace

    @property
    def unembed(self) -> Unembed:
        inputs = self.ln_f.scale * self.resid[-1]
        return Unembed(W_U=self.W_U, inputs=inputs, final_layer=len(self.blocks))

    @property
    def logits(self) -> Float[torch.Tensor, "seq vocab"]:
        return self.ln_f.output @ self.W_U.T

    @property
    def frozen_norms(self) -> FrozenNorms:
        return FrozenNorms(
            ln_1=[block.ln_1.stats for block in self.blocks],
            ln_2=[block.ln_2.stats for block in self.blocks],
            ln_f=self.ln_f.stats,
        )


def layer_norm(x: Stream, ln: nn.LayerNorm, stats: NormStats | None) -> LayerNormTrace:
    if stats is None:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        stats = NormStats(
            mean=x.mean(dim=-1, keepdim=True), std=(variance + ln.eps).sqrt()
        )
    scale = ln.weight / stats.std
    output = (x - stats.mean) * scale + ln.bias
    return LayerNormTrace(stats=stats, scale=scale, output=output)


def blocked_sources(
    seq: int, attention_type: str, window: int
) -> Bool[torch.Tensor, "seq src"]:
    pos = torch.arange(seq)[:, None]
    src = torch.arange(seq)[None, :]
    future = src > pos
    if attention_type == "local":
        return future | (pos - src >= window)
    return future


def trace_attn(
    attn: GPTNeoSelfAttention, ln_out: Stream, keep: AttnLayerKeep
) -> AttnLayerTrace:
    """GPT-Neo does not scale its scores by 1/sqrt(head_dim), so `scaling` is 1."""
    heads, head_dim = attn.num_heads, attn.head_dim
    W_Q = per_head(attn.q_proj.weight, heads)
    W_K = per_head(attn.k_proj.weight, heads)
    W_V = per_head(attn.v_proj.weight, heads)
    W_O = attn.out_proj.weight.view(-1, heads, head_dim).transpose(0, 1)

    q = torch.einsum("hjd,sd->hsj", W_Q, ln_out) * keep.q
    k = torch.einsum("hjd,sd->hsj", W_K, ln_out) * keep.k
    v = torch.einsum("hjd,sd->hsj", W_V, ln_out) * keep.v

    blocked = blocked_sources(
        ln_out.shape[0], attn.attention_type, attn.config.window_size
    )
    pattern, z, head_writes = attend(q, k, v, W_O, 1.0, blocked, keep)
    return AttnLayerTrace(
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        scaling=1.0,
        q=q,
        k=k,
        v=v,
        pattern=pattern,
        z=z,
        head_writes=head_writes,
    )


def trace_mlp(
    block: GPTNeoBlock,
    ln_out: Stream,
    keep_hidden: torch.Tensor,
    keep_down: torch.Tensor,
) -> MlpTrace:
    mlp = block.mlp
    pre = ln_out @ mlp.c_fc.weight.T + mlp.c_fc.bias
    hidden = mlp.act(pre) * keep_hidden
    writes = hidden @ mlp.c_proj.weight.T * keep_down
    return MlpTrace(
        W_in=mlp.c_fc.weight, W_out=mlp.c_proj.weight, hidden=hidden, writes=writes
    )


def trace_block(
    block: GPTNeoBlock,
    idx: int,
    resid: Stream,
    keep: GptNeoKeep,
    frozen: FrozenNorms | None,
) -> tuple[BlockTrace, Stream, Stream]:
    attn = block.attn.attention
    ln_1 = layer_norm(resid, block.ln_1, frozen.ln_1[idx] if frozen else None)
    attn_trace = trace_attn(attn, ln_1.output, keep.layers[idx])
    attn_out = attn_trace.head_writes.sum(dim=0) + attn.out_proj.bias
    resid_mid = (resid + attn_out) * keep.resid_mid[idx]

    ln_2 = layer_norm(resid_mid, block.ln_2, frozen.ln_2[idx] if frozen else None)
    mlp = trace_mlp(block, ln_2.output, keep.mlp_hidden[idx], keep.mlp_down[idx])
    resid_next = (resid_mid + mlp.writes + block.mlp.c_proj.bias) * keep.resid[idx + 1]
    trace = BlockTrace(ln_1=ln_1, attn=attn_trace, ln_2=ln_2, mlp=mlp)
    return trace, resid_mid, resid_next


@torch.no_grad()
def trace_gpt_neo(
    model: GPTNeoForCausalLM,
    tokens: Int[torch.Tensor, " seq"],
    keep: GptNeoKeep | None = None,
    frozen: FrozenNorms | None = None,
) -> GptNeoTrace:
    seq = tokens.shape[0]
    keep = keep or keep_all(model.config, seq)
    stack = model.transformer
    token_embed = stack.wte.weight[tokens] * keep.token_embed
    pos_embed = stack.wpe.weight[:seq] * keep.pos_embed
    resid = [(token_embed + pos_embed) * keep.resid[0]]
    resid_mid, blocks = [], []
    for idx, block in enumerate(stack.h):
        block_trace, mid, resid_next = trace_block(
            cast(GPTNeoBlock, block), idx, resid[-1], keep, frozen
        )
        blocks.append(block_trace)
        resid_mid.append(mid)
        resid.append(resid_next)
    ln_f = layer_norm(resid[-1], stack.ln_f, frozen.ln_f if frozen else None)
    return GptNeoTrace(
        tokens=tokens,
        W_U=model.lm_head.weight,
        token_embed=token_embed,
        pos_embed=pos_embed,
        resid=resid,
        resid_mid=resid_mid,
        blocks=blocks,
        ln_f=ln_f,
    )
