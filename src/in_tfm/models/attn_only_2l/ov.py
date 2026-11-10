"""OV attribution for a one-layer attention-only model: the QK circuit is left out.

- The stream after the layer is the embedding plus every head's write, with no norm anywhere.
  - So at the firing position p, residual dim d is `x[p, d] + sum_h sum_t A_h[p, t] * (W_O_h v_h[t])[d]`.
  - With the attention pattern held fixed, that sum is exact and splits over source tokens t.
- A token's relevance is its term in that sum, never a gradient through the softmax.
  - So unlike AttnLRP, nothing says why a head attended where it did.
- What the relevance explains depends on the reported layer:
  - `o_proj.heads.h` neuron d: head h's write to d, one number per source token.
  - `o_proj` neuron d: the same, summed over heads.
  - `lm_head` neuron n: per residual dim, `W_U[n, d]` times its embedding and value terms.
    - The embedding term sits on the firing token alone, since the embedding is its own.
- Biases on `o_proj` and `lm_head` reach no token, so they are left out of the relevance, as in AttnLRP.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import nn

from ...layers import LayerGetter
from ...sources import ModelBatch
from . import model as attn_model
from .model import AttnOnlyForCausalLM, PerHeadOutput

HeadWrites = Float[torch.Tensor, "head seq hidden"]


class AttentionRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Float[torch.Tensor, "batch head seq seq"]
    value: Float[torch.Tensor, "batch head seq head_dim"]


class OVTarget(BaseModel):
    kind: Literal["head", "o_proj", "lm_head"]
    head: int | None = None


@contextmanager
def recorded_attention() -> Iterator[list[AttentionRecord]]:
    """Wraps whatever attention kernel is installed, so an AttnLRP-patched one is recorded too."""
    records: list[AttentionRecord] = []
    installed = attn_model.eager_attention_forward

    def recording(module, query, key, value, attention_mask, scaling, **kwargs):
        out, weights = installed(
            module, query, key, value, attention_mask, scaling, **kwargs
        )
        records.append(AttentionRecord(weights=weights.detach(), value=value.detach()))
        return out, weights

    attn_model.eager_attention_forward = recording
    try:
        yield records
    finally:
        attn_model.eager_attention_forward = installed


def check_one_layer(model: AttnOnlyForCausalLM) -> None:
    if model.config.num_hidden_layers != 1:
        raise ValueError(
            "OV attribution splits a single layer's writes; "
            f"this model has {model.config.num_hidden_layers} layers"
        )


def ov_target(model: AttnOnlyForCausalLM, layer: nn.Module) -> OVTarget:
    o_proj = model.model.layers[0].self_attn.o_proj
    assert isinstance(o_proj, PerHeadOutput)
    if layer is model.lm_head:
        return OVTarget(kind="lm_head")
    if layer is o_proj:
        return OVTarget(kind="o_proj")
    for head, head_layer in enumerate(o_proj.heads):
        if layer is head_layer:
            return OVTarget(kind="head", head=head)
    raise ValueError(
        f"OV attribution explains lm_head, o_proj or one o_proj head, not {type(layer).__name__}"
    )


def head_writes_at(
    record: AttentionRecord, o_proj: PerHeadOutput, position: int
) -> HeadWrites:
    """Each head's write from each source token to the stream at `position`, the attention already applied."""
    weights = record.weights[0, :, position, :]
    writes = [
        record.value[0, head] @ o_proj.head_weight(head).T
        for head in range(len(o_proj.heads))
    ]
    return weights[..., None] * torch.stack(writes)


def stream_input(model: AttnOnlyForCausalLM, batch: ModelBatch) -> torch.Tensor:
    """The layer's input, which is the embedding, captured rather than rebuilt from the batch."""
    captured: dict[str, torch.Tensor] = {}
    handle = model.model.layers[0].register_forward_pre_hook(
        lambda module, args: captured.__setitem__("input", args[0].detach())
    )
    try:
        model(**batch.kwargs)
    finally:
        handle.remove()
    return captured["input"]


def target_relevance(
    target: OVTarget,
    writes: HeadWrites,
    embedding: Float[torch.Tensor, "seq hidden"],
    position: int,
    neuron_idx: int,
    unembed_row: Float[torch.Tensor, "hidden"],
) -> Float[torch.Tensor, "seq k"]:
    match target.kind:
        case "head":
            return writes[target.head, :, neuron_idx, None]
        case "o_proj":
            return writes[:, :, neuron_idx].sum(0)[:, None]
        case "lm_head":
            terms = writes.sum(0)
            terms[position] += embedding[position]
            return terms * unembed_row


def compute_ov_relevance(
    model: nn.Module,
    batch: ModelBatch,
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int | None,
) -> Float[np.ndarray, "batch seq k"]:
    """Same signature as `compute_attnlrp_relevance`, so the neuron report can take either."""
    assert isinstance(model, AttnOnlyForCausalLM)
    check_one_layer(model)
    if token_idx is None:
        raise ValueError("OV attribution explains one firing position, not a sum")
    target = ov_target(model, layer_getter(model))
    model.eval()
    with torch.no_grad(), recorded_attention() as records:
        embedding = stream_input(model, batch)
    o_proj = model.model.layers[0].self_attn.o_proj
    assert isinstance(o_proj, PerHeadOutput)
    with torch.no_grad():
        writes = head_writes_at(records[0], o_proj, token_idx)
        relevance = target_relevance(
            target,
            writes,
            embedding[0],
            token_idx,
            neuron_idx,
            model.lm_head.weight[neuron_idx],
        )
    return relevance[None].float().cpu().numpy()
