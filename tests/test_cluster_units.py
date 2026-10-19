import numpy as np
import pytest
import torch

from in_tfm.activation_ranges import ActivationRanges, RANGES_NAME
from in_tfm.cluster_units import (
    ClusterGroup,
    ClusterLike,
    HitTable,
    MissingActivationsError,
    activations_by_id,
    load_clusters,
    load_singleton_groups,
    median_activation,
    read_hit_table,
    write_hit_table,
)
from in_tfm.report_meta import ClusterMeta, ElbowPickMeta, ReportMeta, write_report_meta

MEANS = {5: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 9: [0.0, 0.0, 1.0]}
COUNTS = {5: 3, 2: 2, 9: 1}


def fake_report(report_dir, with_ranges=True, with_table=True):
    """Cluster 5 hits samples 0, 0, 1; cluster 2 hits samples 1, 2; cluster 9 hits sample 3."""
    labels = np.array([5, 5, 5, 2, 2, 9, -1])
    batch_idx = np.array([0, 0, 1, 1, 2, 3, 3])
    activations = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 7.0, 0.5])
    write_report_meta(report_dir, ReportMeta(
        neuron_idx=0, pick=ElbowPickMeta(cutoff=0.4), n_hits=len(labels), min_uniq_images_per_cluster=1,
        hadamard_shape=(1, 3), sample_ids=[str(i) for i in range(4)],
        clusters={cid: ClusterMeta(n_hits=COUNTS[cid], n_unique_images=len(set(batch_idx[labels == cid]))) for cid in MEANS},
    ))
    for cid, mean in MEANS.items():
        (report_dir / f"cluster_{cid}").mkdir()
        torch.save(torch.tensor(mean), report_dir / f"cluster_{cid}" / "mean.pt")
    if with_ranges:
        ranges = ActivationRanges(clusters={cid: activations[labels == cid].tolist() for cid in MEANS}, noise=[0.0])
        (report_dir / RANGES_NAME).write_text(ranges.model_dump_json())
    if with_table:
        write_hit_table(report_dir, HitTable(batch_idx=batch_idx, token_idx=np.zeros(7, dtype=int), labels=labels, activations=activations))
    return report_dir


def test_clusters_load_in_the_reports_order_with_their_activations(tmp_path):
    clusters = load_clusters(fake_report(tmp_path))
    assert [c.id for c in clusters] == [5, 2, 9]
    assert clusters[0].activations().tolist() == [1.0, 2.0, 3.0]
    assert median_activation(clusters[1]) == 15.0


def test_clusters_and_groups_both_satisfy_the_protocol(tmp_path):
    groups = load_singleton_groups(fake_report(tmp_path))
    assert all(isinstance(unit, ClusterLike) for unit in [*groups, *groups[0].clusters])


def test_without_a_grouping_every_cluster_is_its_own_group(tmp_path):
    groups = load_singleton_groups(fake_report(tmp_path))
    assert [g.cluster_ids for g in groups] == [[5], [2], [9]]
    assert [g.id for g in groups] == [0, 1, 2]
    torch.testing.assert_close(groups[1].mean_hadamard(), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))


def test_a_groups_mean_is_weighted_by_hits_so_it_is_the_mean_over_all_its_hits(tmp_path):
    clusters = load_clusters(fake_report(tmp_path))
    group = ClusterGroup(id=0, clusters=clusters[:2])
    torch.testing.assert_close(group.mean_hadamard(), torch.tensor([0.6, 0.4, 0.0], dtype=torch.float64))
    assert group.n_hits == 5
    assert sorted(group.activations().tolist()) == [1.0, 2.0, 3.0, 10.0, 20.0]


def test_a_groups_unique_samples_counts_samples_its_clusters_share_once(tmp_path):
    report = fake_report(tmp_path)
    group = ClusterGroup(id=0, clusters=load_clusters(report)[:2], hit_table=read_hit_table(report))
    assert sum(c.n_unique_samples for c in group.clusters) == 4
    assert group.n_unique_samples == 3


def test_a_group_of_several_clusters_has_no_unique_count_without_the_hit_table(tmp_path):
    report = fake_report(tmp_path, with_table=False)
    clusters = load_clusters(report)
    assert ClusterGroup(id=0, clusters=clusters[:2]).n_unique_samples is None
    assert ClusterGroup(id=0, clusters=clusters[:1]).n_unique_samples == 2


def test_a_report_without_ranges_says_so_when_activations_are_asked_for(tmp_path):
    cluster = load_clusters(fake_report(tmp_path, with_ranges=False))[0]
    with pytest.raises(MissingActivationsError):
        cluster.activations()


def test_activations_by_id_is_what_the_range_plot_takes(tmp_path):
    groups = load_singleton_groups(fake_report(tmp_path))
    assert activations_by_id(groups) == {0: [1.0, 2.0, 3.0], 1: [10.0, 20.0], 2: [7.0]}


def test_member_cosines_are_one_for_a_single_cluster(tmp_path):
    group = load_singleton_groups(fake_report(tmp_path))[0]
    assert group.member_cosines().tolist() == pytest.approx([1.0])
