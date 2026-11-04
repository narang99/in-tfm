"""A linear layer reading a conv's flattened output, through an elementwise op.

- `nn.Flatten` of a `(channels, h, w)` map puts dim `c * h * w + y * w + x` at channel c, cell (y, x).
- `ConvLayout` numbers the cell `y * w + x`, so a dim is channel `dim // cells` at token `dim % cells`.
- Every dim is then exactly one conv neuron at one position, so each dependency has one writer.
"""

from collections.abc import Callable

import torch

from .dependency_map import Captures, Tap
from .refs import Dependency, NeuronWriter, WriterTerm

Elementwise = Callable[[torch.Tensor], torch.Tensor]


class FlattenMap:
    def __init__(
        self,
        upstream_layer: str,
        n_channels: int,
        n_cells: int,
        target_in_features: int,
        elementwise: Elementwise,
    ) -> None:
        if n_channels * n_cells != target_in_features:
            raise ValueError(
                f"{upstream_layer} has {n_channels} channels x {n_cells} cells, "
                f"but the target reads {target_in_features} dims"
            )
        self.upstream_layer = upstream_layer
        self.upstream = Tap(layer=upstream_layer)
        self.n_cells = n_cells
        self.elementwise = elementwise

    @property
    def taps(self) -> list[Tap]:
        return [self.upstream]

    def dependency(self, dim: int, target_token: int, captures: Captures) -> Dependency:
        """`target_token` is unused: the target reads a whole image as one position."""
        channel, cell = divmod(dim, self.n_cells)
        value = captures[self.upstream][cell, channel]
        writer = NeuronWriter(layer=self.upstream_layer, neuron=channel, token=cell)
        return Dependency(
            dim=dim,
            terms=[
                WriterTerm(
                    writer=writer,
                    value=float(value),
                    term=float(self.elementwise(value)),
                )
            ],
        )
