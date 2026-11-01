"""AttnLRP relevance on every OV dot product of the attention-only model.

- An OV unit is one output of `v_proj` or `o_proj` at one position. Each is one dot product.
- The model must be patched with `patch_attn_only_for_attn_lrp` first.
  - With the patch, activation * gradient at any tensor is that tensor's AttnLRP relevance.
  - So one backward pass from a scalar gives the relevance at every hooked unit and at the input.
- The input leaf is the token embeddings.
  - Position embeddings and biases are parameters, not leaves. Their relevance is the gap between the target and the
    summed token relevance.
"""

from collections.abc import Callable
from typing import Literal

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel

from .model import AttnOnlyForCausalLM

Projection = Literal["v", "o"]
UnitKey = tuple[int, Projection]


class OvUnit(BaseModel):
    layer: int
    projection: Projection
    neuron: int
    position: int


class OvCapture(BaseModel):
    """Each projection's input and output, keyed by (layer, projection), for a batch of one."""

    model_config = {"arbitrary_types_allowed": True}

    inputs: dict[UnitKey, Float[torch.Tensor, "seq in_features"]]
    outputs: dict[UnitKey, Float[torch.Tensor, "seq out_features"]]
    logits: Float[torch.Tensor, "seq vocab"]


class Relevance(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    target: float
    units: dict[UnitKey, Float[torch.Tensor, "seq out_features"]]
    tokens: Float[torch.Tensor, "seq"]


def projection(model: AttnOnlyForCausalLM, key: UnitKey) -> torch.nn.Linear:
    layer, name = key
    return getattr(model.model.layers[layer].self_attn, f"{name}_proj")


def unit_keys(model: AttnOnlyForCausalLM) -> list[UnitKey]:
    return [(layer, name) for layer in range(model.config.num_hidden_layers) for name in ("v", "o")]


def capture_ov(model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"]) -> OvCapture:
    inputs: dict[UnitKey, torch.Tensor] = {}
    outputs: dict[UnitKey, torch.Tensor] = {}

    def record(key: UnitKey):
        def hook(_module, inp, out):
            inputs[key], outputs[key] = inp[0][0].detach(), out[0].detach()
        return hook

    handles = [projection(model, key).register_forward_hook(record(key)) for key in unit_keys(model)]
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    return OvCapture(inputs=inputs, outputs=outputs, logits=logits)


def logit_relevance(model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"], token: int) -> Relevance:
    """Relevance of the last position's logit for `token`."""
    return _relevance_from(model, input_ids, lambda logits, _outputs: logits[0, -1, token])


def unit_relevance(model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"], unit: OvUnit) -> Relevance:
    key = (unit.layer, unit.projection)
    return _relevance_from(model, input_ids, lambda _logits, outputs: outputs[key][0, unit.position, unit.neuron])


def _relevance_from(
    model: AttnOnlyForCausalLM,
    input_ids: Int[torch.Tensor, "1 seq"],
    target_of: Callable[[torch.Tensor, dict[UnitKey, torch.Tensor]], torch.Tensor],
) -> Relevance:
    """`target_of` picks the scalar to explain from the logits and the hooked projection outputs."""
    model.eval()
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_(True)
    outputs: dict[UnitKey, torch.Tensor] = {}

    def record(key: UnitKey):
        def hook(_module, _inp, out):
            out.retain_grad()
            outputs[key] = out
        return hook

    handles = [projection(model, key).register_forward_hook(record(key)) for key in unit_keys(model)]
    try:
        with torch.enable_grad():
            target = target_of(model(inputs_embeds=embeds).logits, outputs)
            target.backward()
    finally:
        for handle in handles:
            handle.remove()
    model.zero_grad(set_to_none=True)
    return Relevance(
        target=target.item(),
        units={key: _grad_times_input(out) for key, out in outputs.items()},
        tokens=_grad_times_input(embeds).sum(dim=-1),
    )


def _grad_times_input(tensor: torch.Tensor) -> torch.Tensor:
    """A unit downstream of the target gets no gradient, so its relevance is zero."""
    grad = tensor.grad if tensor.grad is not None else torch.zeros_like(tensor)
    return (tensor * grad)[0].detach()
