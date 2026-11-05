"""A layer reading the residual stream, where every dim is a sum of writers.

- Each layer that writes to the stream adds its output, so stream dim d at token t is the sum of
  every writer's output dim d at token t.
- A writer with neuron reports is a `ReportedSource`: its dim d is its own neuron d, with clusters.
- A writer without them, like the embedding, is a `FixedSource`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .dependency_map import Captures
from .refs import Dependency, FixedWriter, NeuronWriter, Tap, Writer, WriterTerm


class ReportedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["reported"] = "reported"
    layer: str

    @property
    def tap(self) -> Tap:
        return Tap(layer=self.layer)

    def writer(self, dim: int, token: int) -> Writer:
        return NeuronWriter(layer=self.layer, neuron=dim, token=token)


class FixedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["fixed"] = "fixed"
    name: str
    tap: Tap
    by_token: bool = False

    def writer(self, dim: int, token: int) -> Writer:
        return FixedWriter(
            name=self.name, tap=self.tap, dim=dim, token=token, by_token=self.by_token
        )


StreamSource = Annotated[ReportedSource | FixedSource, Field(discriminator="kind")]


class ResidualStreamMap:
    def __init__(self, sources: list[StreamSource]) -> None:
        self.sources = sources

    @property
    def taps(self) -> list[Tap]:
        return [source.tap for source in self.sources]

    def dependency(self, dim: int, target_token: int, captures: Captures) -> Dependency:
        return Dependency(
            dim=dim,
            terms=[
                self._term(source, dim, target_token, captures)
                for source in self.sources
            ],
        )

    def _term(
        self, source: StreamSource, dim: int, token: int, captures: Captures
    ) -> WriterTerm:
        value = float(captures[source.tap][token, dim])
        return WriterTerm(writer=source.writer(dim, token), value=value, term=value)
