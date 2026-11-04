from pathlib import Path

import numpy as np
import pytest
import torch

from in_tfm.clustered_hits import clustered_hits, write_clustered_hits
from in_tfm.dependencies import NeuronWriter
from in_tfm.dependencies.collect import collect_cluster_dependencies
from in_tfm.dependencies.firings import ClusterMatch, FiringSelection, WriterFiring
from in_tfm.dependencies.matching import WriterMatcher, WriterReport, build_matcher
from in_tfm.dependencies.stats import cluster_shares
from in_tfm.report_meta import ClusterMeta, ReportMeta, write_report_meta
from in_tfm.run_config import ClusteringConfig

WEIGHT = torch.tensor([[1.0, 1.0], [2.0, -1.0]])


def report(distance: str = "euclidean", normalise: bool = True) -> WriterReport:
    return WriterReport(
        layer="up",
        neuron=1,
        report_dir=Path("r"),
        clustering=ClusteringConfig(distance=distance, normalise=normalise),
    )


@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
def test_a_firing_takes_the_cluster_of_its_nearest_clustered_hit(distance):
    references = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    matcher = WriterMatcher(report(distance), WEIGHT, references, np.array([4, 7, 9]))
    (near_second, near_first) = matcher.match(torch.tensor([[0.1, 2.0], [3.0, 0.2]]))
    assert near_second.cluster_id == 7
    assert near_first.cluster_id == 4


def test_normalised_euclidean_ignores_a_firings_size():
    references = torch.tensor([[1.0, 0.0], [10.0, 10.0]])
    matcher = WriterMatcher(report(), WEIGHT, references, np.array([0, 1]))
    (match,) = matcher.match(torch.tensor([[2.0, 2.0]]))
    assert match.cluster_id == 1
    assert match.distance == pytest.approx(0.0, abs=1e-6)


def firing_with(cluster_id: int, distance: float) -> WriterFiring:
    return WriterFiring(
        writer=NeuronWriter(layer="up", neuron=0, token=0),
        value=1.0,
        term=1.0,
        match=ClusterMatch(cluster_id=cluster_id, distance=distance),
    )


def test_a_cluster_with_few_of_a_writers_matches_is_an_outlier():
    writers = [firing_with(0, 0.1)] * 9 + [firing_with(3, 0.5)]
    shares = cluster_shares(writers, outlier_ratio=0.2)
    assert [(c.cluster_id, c.count, c.is_outlier) for c in shares] == [
        (0, 9, False),
        (3, 1, True),
    ]


def write_fake_conv2_report(report_dir: Path, sample_ids: list[str]) -> None:
    """Two clusters of conv2 neuron 0 hits, at made-up positions of the fake images."""
    report_dir.mkdir(parents=True)
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    write_clustered_hits(
        report_dir,
        clustered_hits(
            sample_ids,
            batch_idx=np.array([0, 1, 2, 3, 4, 5, 0]),
            token_idx=np.array([3, 4, 5, 20, 21, 22, 0]),
            labels=labels,
            activations=np.ones(7),
        ),
    )
    write_report_meta(
        report_dir,
        ReportMeta(
            neuron_idx=0,
            picks=[],
            n_hits=7,
            min_uniq_images_per_cluster=1,
            hadamard_shape=(2, 9),
            clusters={
                0: ClusterMeta(n_hits=3, n_unique_images=3),
                1: ClusterMeta(n_hits=3, n_unique_images=3),
            },
            sample_ids=sample_ids,
        ),
    )


def test_mnist_conv2_firings_are_matched_to_the_writer_reports_clusters(
    mnist_origin, tmp_path
):
    setup, origin, adapter = mnist_origin
    model = adapter.model
    writer_dir = tmp_path / "conv2" / "neuron_0"
    write_fake_conv2_report(writer_dir, list(setup.source.sample_ids()))
    conv_weight = adapter.layout_for(model.conv2).weight(model.conv2.weight).detach()
    matcher = build_matcher(
        setup,
        WriterReport(
            layer="conv2",
            neuron=0,
            report_dir=writer_dir,
            clustering=ClusteringConfig(),
        ),
        conv_weight,
    )
    collected = collect_cluster_dependencies(
        setup,
        origin,
        model.head[1].weight[origin.neuron].detach(),
        adapter.dependency_map_for("head.1"),
        FiringSelection(dim_mass=0.9, writer_mass=0.9),
        matchers={("conv2", 0): matcher},
    )
    matched = [w for f in collected.firings for w in f.writers if w.writer.neuron == 0]
    unmatched = [
        w for f in collected.firings for w in f.writers if w.writer.neuron != 0
    ]
    assert matched and all(w.match is not None for w in matched)
    assert all(w.match is None for w in unmatched)
    assert {w.match.cluster_id for w in matched} <= {0, 1}
    assert [r.layer for r in collected.writer_reports] == ["conv2"]
