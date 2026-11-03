"""Picks out one submodule of a model by its dotted path, so callers can act on "the layer of
interest" without hardcoding its attribute path.

A getter works on either the raw model or its NNsight-wrapped tracer - plain attribute/index
access forwards identically through both - so the same getter serves capture and attribution.
"""

from collections.abc import Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.base import NNsight

LayerGetter = Callable[[NNsight | torch.nn.Module], Envoy | torch.nn.Module]


def submodule_getter(layer_name: str) -> LayerGetter:
    """For example `layers.11.mlp.down_proj` or `mixed4e_1x1_pre_relu_conv`.

    - The paths are submodule names of the adapter's `get_model()`, which `named_modules()` lists.
    """

    def getter(model: NNsight | torch.nn.Module) -> Envoy | torch.nn.Module:
        layer = model
        for part in layer_name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        return layer

    return getter
