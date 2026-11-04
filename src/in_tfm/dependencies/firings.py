"""What fired under one example of an origin cluster.

- An example is one hit of the origin neuron, a sample and the token it fired at.
- The target's activation there is a sum over input dims of `input[dim] * weight[dim]`, its Hadamard row.
  - The dims covering `dim_mass` of that row's |value| are the example's dependency firings.
- Each firing's dim is a sum of writer terms.
  - The writers covering `writer_mass` of the terms' |value| are kept on the firing.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel

from ..sources import SampleId
from .consistency import check_terms_rebuild_input
from .dependency_map import Captures, DependencyMap
from .refs import Dependency, Tap, Writer
from .selection import covering_indices


class Example(BaseModel):
    sample_id: SampleId
    token: int


class ClusterMatch(BaseModel):
    """The writer's own cluster a firing looks like, see `matching`."""

    cluster_id: int
    distance: float
    hadamard: list[float]
    """The firing's own Hadamard row on the writer."""
    closest_hadamard: list[float]
    """The nearest clustered hit's, for drawing the two side by side."""


class WriterReportRef(BaseModel):
    """A writer neuron's own report, as the dependency page links to it and draws its rows."""

    layer: str
    neuron: int
    report_dir: Path
    tile_shape: tuple[int, ...]
    """How to draw its Hadamard row: `(channels, k, k)` for a conv, `(dim,)` otherwise."""


@runtime_checkable
class Matcher(Protocol):
    """Matches firings of one writer neuron to its own clusters, see `matching.WriterMatcher`."""

    @property
    def report(self) -> WriterReportRef: ...

    @property
    def tap(self) -> Tap:
        """The writer's input, which its Hadamard rows are built from."""
        ...

    def match(self, inputs: Float[torch.Tensor, "n hidden"]) -> list[ClusterMatch]: ...


class WriterFiring(BaseModel):
    writer: Writer
    value: float
    """In the writer's own units, the ones its noise sample is in."""
    term: float
    """The part of the dim's input value this writer accounts for."""
    match: ClusterMatch | None = None
    """None for a writer with no report to match against."""


def writer_contribution(firing: "DependencyFiring", writer: WriterFiring) -> float:
    """This writer's share of the target's activation. Its sign says whether the writer adds or inhibits."""
    return writer.term * firing.weight


class DependencyFiring(BaseModel):
    example: int
    """Index into the run's examples."""
    dim: int
    input_value: float
    """The target's input on this dim."""
    weight: float
    """The target neuron's weight on this dim."""
    contribution: float
    """`input_value * weight`. Its sign says whether it adds to the target or inhibits it."""
    writers: list[WriterFiring]
    """The writers covering `writer_mass` of the dim, largest |term| first."""
    n_writers: int
    """All of the dim's writers, before any were left out."""


class FiringSelection(BaseModel):
    dim_mass: float
    writer_mass: float


def kept_writers(dependency: Dependency, writer_mass: float) -> list[WriterFiring]:
    terms = np.array([t.term for t in dependency.terms])
    return [
        WriterFiring(
            writer=dependency.terms[i].writer,
            value=dependency.terms[i].value,
            term=dependency.terms[i].term,
        )
        for i in covering_indices(terms, writer_mass)
    ]


def example_firings(
    example: int,
    target_token: int,
    target_input: Float[torch.Tensor, "hidden"],
    weight_row: Float[torch.Tensor, "hidden"],
    captures: Captures,
    dependency_map: DependencyMap,
    selection: FiringSelection,
) -> list[DependencyFiring]:
    """Checks the map rebuilds every input dim, not only the kept ones, so a wrong map fails on any example."""
    dependencies = [
        dependency_map.dependency(dim, target_token, captures)
        for dim in range(len(target_input))
    ]
    check_terms_rebuild_input(dependencies, target_input)
    hadamard = (target_input * weight_row).double().numpy()
    return [
        DependencyFiring(
            example=example,
            dim=int(dim),
            input_value=float(target_input[dim]),
            weight=float(weight_row[dim]),
            contribution=float(hadamard[dim]),
            writers=kept_writers(dependencies[dim], selection.writer_mass),
            n_writers=len(dependencies[dim].terms),
        )
        for dim in covering_indices(hadamard, selection.dim_mass)
    ]
