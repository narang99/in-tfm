"""The model-specific part of a dependency report.

- Only a model's adapter knows how a target layer's input is built from the layers below it.
- An adapter that knows it implements `SupportsDependencies`. One that does not cannot make dependency reports.
- A map reads writers from taps: a layer's input or output, saved in the same forward pass as the target.
  - An input tap is for values no module outputs, like the embeddings a text source passes in.
"""

from typing import Literal, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

from .refs import Dependency


class Tap(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer: str
    port: Literal["input", "output"] = "output"


Captures = dict[Tap, Float[torch.Tensor, "seq hidden"]]
"""One sample's tapped tensors, each in its layer's token layout."""


@runtime_checkable
class DependencyMap(Protocol):
    @property
    def taps(self) -> list[Tap]: ...

    def dependency(
        self, dim: int, target_token: int, captures: Captures
    ) -> Dependency: ...


@runtime_checkable
class SupportsDependencies(Protocol):
    def dependency_map_for(self, target_layer: str) -> DependencyMap:
        """Raises `ValueError` for a target layer whose input the adapter cannot map."""
        ...
