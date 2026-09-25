"""Callables that pick out one submodule of a Dinov2 model, so callers can act on "the layer
of interest" without hardcoding its attribute path.

A getter works on either the raw model or its NNsight-wrapped tracer - plain attribute/index
access forwards identically through both - so the same getter serves activations.py's tracing
and attribution.py's direct forward/backward passes.
"""

from collections.abc import Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.base import NNsight
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

LayerGetter = Callable[[NNsight | torch.nn.Module], Envoy | torch.nn.Module]


def fc2_getter(layer_idx: int) -> LayerGetter:
    def getter(model: NNsight | Dinov2Model) -> Envoy | torch.nn.Module:
        return model.encoder.layer[layer_idx].mlp.fc2

    return getter


def down_proj_getter(layer_idx: int) -> LayerGetter:
    """The decoder-LM analogue of fc2: the projection back from the MLP's intermediate width to
    the residual stream, so its *input* carries the same "which intermediate neurons fired"
    signal the Hadamard product decomposes.

    Named for Gemma/Llama/Qwen, which all call it `down_proj`.
    """

    def getter(model: NNsight | torch.nn.Module) -> Envoy | torch.nn.Module:
        return model.layers[layer_idx].mlp.down_proj

    return getter


def q_proj_getter(layer_idx: int) -> LayerGetter:
    """The query projection, whose output is (batch, seq, n_heads * head_dim) with head `h`
    owning columns `h * head_dim : (h + 1) * head_dim`. A flat neuron index is therefore
    `h * head_dim + d`.

    Deliberately the *raw* q_proj output, before `q_norm` and rope:
    - it is linear in the input, so input * weight[neuron] is an exact decomposition of the
      activation over residual dimensions, as it is for down_proj;
    - rope has not yet mixed in the token's position, so the coordinate is position-free.
    """

    def getter(model: NNsight | torch.nn.Module) -> Envoy | torch.nn.Module:
        return model.layers[layer_idx].self_attn.q_proj

    return getter


def k_proj_getter(layer_idx: int) -> LayerGetter:
    """The key projection, the counterpart of `q_proj_getter`. Its output is
    (batch, seq, n_kv_heads * head_dim), so a flat neuron index is `kv_head * head_dim + d`.

    Two differences from q_proj to keep in mind when reading a report:
    - Gemma 3 uses grouped-query attention, so one kv head serves `n_heads // n_kv_heads` query
      heads. A key neuron is therefore shared by several query heads' scores.
    - Like q_proj this is the raw output, before `k_norm` and rope.
    """

    def getter(model: NNsight | torch.nn.Module) -> Envoy | torch.nn.Module:
        return model.layers[layer_idx].self_attn.k_proj

    return getter
