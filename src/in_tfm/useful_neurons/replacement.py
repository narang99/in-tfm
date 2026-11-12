"""What a dropped neuron becomes instead of its activation.

The trace asks a `Replace` callable for the numbers at a site, one kind of neuron in one layer.
The callable only maps sites to numbers. Which numbers, zeros or stored means, is up to it.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    """One kind of neuron in one layer, named like its `Keep` field. Embeddings have no layer."""

    model_config = ConfigDict(frozen=True)

    name: str
    layer: int | None = None

    @property
    def key(self) -> str:
        return self.name if self.layer is None else f"{self.name}/{self.layer}"


Replace = Callable[[Site], torch.Tensor]
"""Returns a tensor that broadcasts against the site's activation, seq axis of size 1."""

LayerFill = Callable[[str], torch.Tensor]


def zeros(site: Site) -> torch.Tensor:
    return torch.tensor(0.0)


class PersistedValues(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: dict[str, torch.Tensor]

    def __call__(self, site: Site) -> torch.Tensor:
        return self.values[site.key]

    def save(self, path: Path) -> None:
        torch.save(self.values, path)

    @classmethod
    def load(cls, path: Path) -> "PersistedValues":
        return cls(values=torch.load(path, weights_only=True))


def at_layer(replace: Replace, layer: int) -> LayerFill:
    return lambda name: replace(Site(name=name, layer=layer))


def keep_or_fill(
    activation: torch.Tensor, keep: torch.Tensor, fill: torch.Tensor
) -> torch.Tensor:
    return torch.where(keep, activation, fill.to(activation.dtype))
