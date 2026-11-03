"""Picks out one submodule of a model by its dotted path, so callers can act on "the layer of
interest" without hardcoding its attribute path.

A getter works on either the raw model or its NNsight-wrapped tracer - plain attribute/index
access forwards identically through both - so the same getter serves capture and attribution.
"""

from typing import Any, Protocol, overload, runtime_checkable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.base import NNsight


@runtime_checkable
class LayerGetter(Protocol):
    """Overloaded, so a raw model gives back a module and a wrapped model gives back an `Envoy`."""

    @overload
    def __call__(self, model: NNsight) -> Envoy: ...
    @overload
    def __call__(self, model: torch.nn.Module) -> torch.nn.Module: ...


class SubmoduleGetter:
    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name

    @overload
    def __call__(self, model: NNsight) -> Envoy: ...
    @overload
    def __call__(self, model: torch.nn.Module) -> torch.nn.Module: ...
    def __call__(self, model: NNsight | torch.nn.Module) -> Envoy | torch.nn.Module:
        layer: Any = model
        for part in self.layer_name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        return layer


def submodule_getter(layer_name: str) -> LayerGetter:
    """For example `layers.11.mlp.down_proj` or `mixed4e_1x1_pre_relu_conv`.

    - The paths are submodule names of the adapter's `get_model()`, which `named_modules()` lists.
    """
    return SubmoduleGetter(layer_name)
