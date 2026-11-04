import numpy as np
import torch

from in_tfm.dependencies import Tap, TapColumn
from in_tfm.dependencies.collect import (
    collect_cluster_dependencies,
    load_cluster_dependencies,
    write_cluster_dependencies,
)
from in_tfm.dependencies.firings import FiringSelection
from in_tfm.dependencies.noise import NoiseSampler
from in_tfm.dependencies.selection import covering_indices


def test_both_signs_are_ranked_together_until_the_mass_is_covered():
    values = np.array([1.0, -6.0, 3.0, 0.0])
    np.testing.assert_array_equal(covering_indices(values, 0.6), [1])
    np.testing.assert_array_equal(covering_indices(values, 0.7), [1, 2])
    np.testing.assert_array_equal(covering_indices(values, 1.0), [1, 2, 0])


def test_an_all_zero_row_covers_nothing():
    assert len(covering_indices(np.zeros(3), 0.9)) == 0


def test_noise_keeps_about_its_size_from_valid_positions_only():
    tap = Tap(layer="layer")
    column = TapColumn(tap=tap, column=1)
    sampler = NoiseSampler([column], n_samples=400, seed=0, size=100)
    valid = torch.ones(400, 10, dtype=torch.bool)
    valid[:, 5:] = False
    values = torch.zeros(400, 10, 2)
    values[:, 5:] = 99.0
    sampler.add({tap: values}, valid)
    (sample,) = sampler.samples()
    assert sample.column == column
    assert 25 < len(sample.values) < 75
    assert max(sample.values) == 0.0


def test_an_mnist_cluster_collects_each_examples_covering_dims(mnist_origin):
    setup, origin, adapter = mnist_origin
    weight_row = adapter.model.head[1].weight[origin.neuron].detach()
    collected = collect_cluster_dependencies(
        setup,
        origin,
        weight_row,
        adapter.dependency_map_for("head.1"),
        FiringSelection(dim_mass=0.9, writer_mass=0.9),
        seed=0,
    )
    assert [e.sample_id for e in collected.examples] == ["test:0", "test:1", "test:3"]
    for example in range(3):
        firings = [f for f in collected.firings if f.example == example]
        dims = [f.dim for f in firings]
        assert len(dims) == len(set(dims))
        assert all(len(f.writers) == f.n_writers == 1 for f in firings)

    columns = {sample.column for sample in collected.noise}
    fired_writers = {w.writer.column for f in collected.firings for w in f.writers}
    assert fired_writers <= columns
    assert (
        TapColumn(
            tap=Tap(layer="head.1", port="input"), column=collected.firings[0].dim
        )
        in columns
    )


def test_collected_dependencies_round_trip(mnist_origin):
    setup, origin, adapter = mnist_origin
    collected = collect_cluster_dependencies(
        setup,
        origin,
        adapter.model.head[1].weight[origin.neuron].detach(),
        adapter.dependency_map_for("head.1"),
        FiringSelection(dim_mass=0.9, writer_mass=0.9),
        seed=0,
    )
    write_cluster_dependencies(collected)
    assert load_cluster_dependencies(origin.report_dir, origin.cluster_id) == collected
