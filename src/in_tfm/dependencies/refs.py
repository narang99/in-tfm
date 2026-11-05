"""What one input dim of a target layer is made of.

- A target neuron's activation is a sum over its input dims of `input[dim] * weight[neuron, dim]`.
- Each input dim is itself made of one or more writers, upstream values at known positions.
  - A dim fed through an elementwise op, like a ReLU after a conv, has one writer.
  - A residual stream dim has many, one per layer that writes to the stream, plus the embedding.
- A writer's `value` is in its own units, the ones its neuron report and noise use.
- A writer's `term` is the part of `input[dim]` it accounts for, so a dim's terms sum to it.
  - For a ReLU'd writer the term is `relu(value)`.
  - For a residual writer the term is the value itself.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Tap(BaseModel):
    """A layer's input or output, saved in the same forward pass as the target.

    - An input tap is for values no module outputs, like the embeddings a text source passes in.
    """

    model_config = ConfigDict(frozen=True)

    layer: str
    port: Literal["input", "output"] = "output"


class TapColumn(BaseModel):
    """One neuron or dim of a tap at any token, the unit a noise sample is drawn for."""

    model_config = ConfigDict(frozen=True)

    tap: Tap
    column: int


class NeuronWriter(BaseModel):
    """A neuron with its own report, so a firing of it can be matched to its clusters."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["neuron"] = "neuron"
    layer: str
    neuron: int
    token: int

    @property
    def column(self) -> TapColumn:
        return TapColumn(tap=Tap(layer=self.layer), column=self.neuron)


class FixedWriter(BaseModel):
    """A term with no clusters, like a token embedding or a bias."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["fixed"] = "fixed"
    name: str
    tap: Tap
    dim: int
    token: int
    by_token: bool = False
    """A token embedding is fixed per token, so its "cluster" is the token id itself."""

    @property
    def column(self) -> TapColumn:
        return TapColumn(tap=self.tap, column=self.dim)


Writer = Annotated[NeuronWriter | FixedWriter, Field(discriminator="kind")]


class WriterTerm(BaseModel):
    writer: Writer
    value: float
    term: float


class Dependency(BaseModel):
    dim: int
    """The target input dim, which is the dependency's identity across examples."""
    terms: list[WriterTerm]

    @property
    def total(self) -> float:
        return sum(t.term for t in self.terms)
