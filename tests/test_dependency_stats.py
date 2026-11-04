import numpy as np
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
from in_tfm.dependencies.stats import (
    StatsSettings,
    check_one_firing_per_dim,
    dependency_stats,
    frequent_threshold,
    noise_stats,
    shorth,
)

TARGET = Tap(layer="head", port="input")


def hp_do_shorth(data, frac):
    """hp-do's loop, kept to check the vectorised port against it."""
    data_sorted = np.sort(data)
    n = len(data_sorted)
    window = int(np.ceil(frac * n))
    min_width, best_start = np.inf, 0
    for i in range(n - window + 1):
        width = data_sorted[i + window - 1] - data_sorted[i]
        if width < min_width:
            min_width, best_start = width, i
    return data_sorted[best_start], data_sorted[best_start + window - 1]


def test_shorth_matches_hp_dos_loop():
    rng = np.random.default_rng(0)
    for size in (5, 37, 1000):
        values = rng.standard_cauchy(size)
        assert shorth(values, 0.9) == pytest.approx(hp_do_shorth(values, 0.9))


def test_a_noise_unit_is_the_reach_of_the_shorth_above_the_median():
    stats = noise_stats(np.array([0.0, 1.0, 2.0, 3.0, 100.0]), 0.6)
    assert (stats.low, stats.median, stats.high) == (0.0, 2.0, 2.0)
    assert stats.units(2.0) == 0.0


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


def test_histograms_pool_writer_firings_of_frequent_dependencies_by_their_own_sign(
    collected,
):
    stats = dependency_stats(collected, StatsSettings(threshold=0.8))
    assert sorted(stats.distances.all) == pytest.approx([-0.5, 0.5, 3.0])
    assert stats.distances.inhibiting == pytest.approx([-0.5])
    assert sorted(stats.fractions_above.adding) == [0.0, 1.0]
    assert stats.fractions_above.inhibiting == [0.0]


def test_a_dim_firing_twice_in_one_example_is_rejected(collected):
    with pytest.raises(ValueError, match="duplicate"):
        check_one_firing_per_dim(collected.firings + collected.firings[:1])
