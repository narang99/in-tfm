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

LayerGetter = Callable[[NNsight | Dinov2Model], Envoy | torch.nn.Module]


def fc2_getter(layer_idx: int) -> LayerGetter:
    def getter(model: NNsight | Dinov2Model) -> Envoy | torch.nn.Module:
        return model.encoder.layer[layer_idx].mlp.fc2

    return getter
