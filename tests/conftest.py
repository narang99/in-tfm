"""Fixtures shared by the dependency stats and page tests."""

import pytest

from in_tfm.dependencies import NeuronWriter, Tap, TapColumn
from in_tfm.dependencies.collect import ClusterDependencies, OriginCluster
from in_tfm.dependencies.firings import (
    DependencyFiring,
    Example,
    FiringSelection,
    WriterFiring,
)
from in_tfm.dependencies.noise import NoiseSample

TARGET = Tap(layer="head", port="input")


def writer(neuron: int, value: float, term: float) -> WriterFiring:
    return WriterFiring(
        writer=NeuronWriter(layer="up", neuron=neuron, token=0), value=value, term=term
    )


def firing(example: int, dim: int, weight: float, writers: list[WriterFiring]):
    total = sum(w.term for w in writers)
    return DependencyFiring(
        example=example,
        dim=dim,
        input_value=total,
        weight=weight,
        contribution=total * weight,
        writers=writers,
        n_writers=2,
    )


def noise_for(column: TapColumn) -> NoiseSample:
    """Three values, so the 90% shorth is all of them: median 0 and a unit of 1."""
    return NoiseSample(column=column, values=[-1.0, 0.0, 1.0])


@pytest.fixture
def collected() -> ClusterDependencies:
    firings = [
        firing(0, 0, 2.0, [writer(0, 3.0, 3.0), writer(1, -0.5, -0.5)]),
        firing(1, 0, 2.0, [writer(0, 0.5, 0.5)]),
        firing(2, 1, -1.0, [writer(1, 2.0, 2.0)]),
    ]
    columns = [TapColumn(tap=Tap(layer="up"), column=n) for n in (0, 1)]
    columns += [TapColumn(tap=TARGET, column=d) for d in (0, 1)]
    return ClusterDependencies(
        origin=OriginCluster(report_dir="r", layer="head", neuron=0, cluster_id=0),
        selection=FiringSelection(dim_mass=0.9, writer_mass=0.9),
        examples=[Example(sample_id=str(i), token=0) for i in range(3)],
        firings=firings,
        noise=[noise_for(c) for c in columns],
    )
