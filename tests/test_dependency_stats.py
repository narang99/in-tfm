import pytest

from in_tfm.dependencies import NeuronWriter
from in_tfm.dependencies.collect import ClusterDependencies, OriginCluster
from in_tfm.dependencies.firings import (
    DependencyFiring,
    Example,
    FiringSelection,
    WriterFiring,
)
from in_tfm.dependencies.stats import (
    StatsSettings,
    check_one_firing_per_dim,
    dependency_stats,
    frequent_threshold,
)


def test_frequent_needs_at_least_two_firings_however_few_examples():
    assert frequent_threshold(5, StatsSettings()) == 2
    assert frequent_threshold(104, StatsSettings()) == 11


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


@pytest.fixture
def collected() -> ClusterDependencies:
    firings = [
        firing(0, 0, 2.0, [writer(0, 3.0, 3.0), writer(1, -0.5, -0.5)]),
        firing(1, 0, 2.0, [writer(0, 0.5, 0.5)]),
        firing(2, 1, -1.0, [writer(1, 2.0, 2.0)]),
    ]
    return ClusterDependencies(
        origin=OriginCluster(report_dir="r", layer="head", neuron=0, cluster_id=0),
        selection=FiringSelection(dim_mass=0.9, writer_mass=0.9),
        examples=[Example(sample_id=str(i), token=0) for i in range(3)],
        firings=firings,
    )


def test_a_dependency_counts_its_examples_and_its_writers_count_its_firings(collected):
    stats = dependency_stats(collected)
    dim0, dim1 = stats.dependencies
    assert (dim0.n_fired, dim0.fired_ratio, dim0.frequent) == (2, 2 / 3, True)
    assert dim1.frequent is False
    by_label = {w.label: w for w in dim0.writers}
    assert by_label["up:0"].fired_ratio == 1.0
    assert by_label["up:1"].fired_ratio == 0.5
    assert by_label["up:1"].median_contribution == -1.0


def test_concentration_ranks_each_sign_by_median_magnitude(collected):
    concentration = dependency_stats(collected).concentration
    assert concentration.positive == [3.0]
    assert concentration.negative == [2.0]
    assert concentration.markers[0].cumulative_share == 1.0


def test_a_dim_firing_twice_in_one_example_is_rejected(collected):
    with pytest.raises(ValueError, match="duplicate"):
        check_one_firing_per_dim(collected.firings + collected.firings[:1])
