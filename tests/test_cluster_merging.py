from conftest import FakeCluster, fake_report
from in_tfm.cluster_merging import grouping_name, merge_clusters, merge_report_clusters
from in_tfm.cluster_merging.strategies.thresholds import threshold_assignment
from in_tfm.cluster_units import load_clusters, load_groups
from in_tfm.report_meta import ReportMeta, load_report_meta, write_report_meta
from in_tfm.run_config import MergeClustersConfig

THRESHOLDS = MergeClustersConfig(method="thresholds", min_cosine=0.98, max_median_gap=0.1)


def cluster(mean, median, n=3, first_sample=0):
    return FakeCluster(mean=mean, activations=[median] * n, samples=list(range(first_sample, first_sample + n)))


def test_a_second_term_at_a_quarter_of_the_peak_splits_two_clusters_cosine_alone_would_merge(tmp_path):
    """Clusters 106 and 109 of the 2/4 MNIST net: cosine 0.974, which looked like a match."""
    report = fake_report(tmp_path, {106: cluster([0.97, -0.22], 1.0), 109: cluster([1.0, 0.0], 1.0, first_sample=3)})
    assert merge_clusters(load_clusters(report), THRESHOLDS, seed=0).members == [[106], [109]]


def test_the_same_direction_at_very_different_sizes_is_split(tmp_path):
    report = fake_report(tmp_path, {0: cluster([1.0, 0.0], 1.0), 1: cluster([1.0, 0.0], 3.0, first_sample=3)})
    assert merge_clusters(load_clusters(report), THRESHOLDS, seed=0).members == [[0], [1]]


def test_close_direction_and_size_merge(tmp_path):
    report = fake_report(tmp_path, {0: cluster([1.0, 0.0], 3.0), 1: cluster([1.0, 0.01], 2.9, first_sample=3)})
    assert merge_clusters(load_clusters(report), THRESHOLDS, seed=0).members == [[0, 1]]


def test_groups_cannot_chain_two_clusters_that_are_too_far_apart(tmp_path):
    """0 and 1 are within 8% of the scale, 1 and 2 too, but 0 and 2 are 16% apart."""
    report = fake_report(tmp_path, {
        0: cluster([1.0, 0.0], 0.84), 1: cluster([1.0, 0.0], 0.92, first_sample=3), 2: cluster([1.0, 0.0], 1.0, first_sample=6),
    })
    members = merge_clusters(load_clusters(report), THRESHOLDS, seed=0).members
    assert len(members) == 2
    assert not any({0, 2} <= set(group) for group in members)


def partition(units, assignment) -> set[frozenset[int]]:
    return {frozenset(u.id for u, label in zip(units, assignment) if label == group) for group in set(assignment)}


def test_the_size_gap_follows_the_neurons_own_scale(tmp_path):
    """Medians 84, 93 and 100 group as {0} and {1, 2} at any scale."""
    clusters = {0: cluster([1.0, 0.0], 84.0), 1: cluster([1.0, 0.0], 93.0, first_sample=3), 2: cluster([1.0, 0.0], 100.0, first_sample=6)}
    small = {cid: cluster(c.mean, c.activations[0] / 100, first_sample=3 * cid) for cid, c in clusters.items()}
    (tmp_path / "big").mkdir()
    (tmp_path / "small").mkdir()
    for folder, given in [("big", clusters), ("small", small)]:
        units = load_clusters(fake_report(tmp_path / folder, given))
        assert partition(units, threshold_assignment(units, 0.98, 0.1)) == {frozenset({0}), frozenset({1, 2})}


def test_groups_are_ordered_largest_first(tmp_path):
    report = fake_report(tmp_path, {0: cluster([1.0, 0.0], 1.0, n=2), 1: cluster([0.0, 1.0], 1.0, n=5, first_sample=2)})
    assert merge_clusters(load_clusters(report), THRESHOLDS, seed=0).members == [[1], [0]]


def test_kmeans_picks_k_from_its_sweep_and_records_it(tmp_path):
    report = fake_report(tmp_path, {
        i: cluster(mean, 1.0, first_sample=3 * i)
        for i, mean in enumerate([[1, 0, 0], [0.99, 0.05, 0], [0, 1, 0], [0.05, 0.99, 0], [0, 0, 1], [0, 0.05, 0.99]])
    })
    grouping = merge_clusters(load_clusters(report), MergeClustersConfig(method="kmeans"), seed=0)
    assert grouping.k == 3
    assert sorted(map(sorted, grouping.members)) == [[0, 1], [2, 3], [4, 5]]
    assert grouping.sweep.ks == [1, 2, 3, 4, 5]


def test_kmeans_with_too_few_clusters_keeps_each_alone(tmp_path):
    report = fake_report(tmp_path, {0: cluster([1.0, 0.0], 1.0), 1: cluster([1.0, 0.0], 1.0, first_sample=3)})
    grouping = merge_clusters(load_clusters(report), MergeClustersConfig(method="kmeans"), seed=0)
    assert grouping.members == [[0], [1]] and grouping.sweep is None


def test_merging_a_report_adds_a_named_default_grouping_that_load_groups_reads(tmp_path):
    report = fake_report(tmp_path, {0: cluster([1.0, 0.0], 3.0), 1: cluster([1.0, 0.01], 2.9, first_sample=3)})
    name = merge_report_clusters(report, THRESHOLDS, seed=0)
    assert name == grouping_name(THRESHOLDS) == "thresholds-cos0.98-gap0.1"
    assert load_report_meta(report).default_grouping == name
    groups = load_groups(report)
    assert [g.cluster_ids for g in groups] == [[0, 1]]
    assert groups[0].n_unique_samples == 6


def test_a_second_grouping_is_added_beside_the_first(tmp_path):
    report = fake_report(tmp_path)
    merge_report_clusters(report, THRESHOLDS, seed=0)
    merge_report_clusters(report, MergeClustersConfig(method="kmeans", k=2), seed=0)
    meta = load_report_meta(report)
    assert set(meta.groupings) == {"thresholds-cos0.98-gap0.1", "kmeans-k2"}
    assert meta.default_grouping == "kmeans-k2"
    assert [g.cluster_ids for g in load_groups(report, "thresholds-cos0.98-gap0.1")] == [[5], [2], [9]]


def test_method_none_leaves_the_report_alone(tmp_path):
    report = fake_report(tmp_path)
    assert merge_report_clusters(report, MergeClustersConfig(), seed=0) is None
    assert load_report_meta(report).groupings == {}


def test_a_report_written_before_groupings_still_loads(tmp_path):
    raw = load_report_meta(fake_report(tmp_path)).model_dump(exclude={"groupings", "default_grouping"})
    assert ReportMeta.model_validate(raw).groupings == {}


def test_a_report_without_clusters_gets_no_grouping(tmp_path):
    """A logit of the 2/4 MNIST net had 0 clusters, and an empty grouping broke its page."""
    report = fake_report(tmp_path)
    meta = load_report_meta(report)
    write_report_meta(report, meta.model_copy(update={"clusters": {}}))
    assert merge_report_clusters(report, THRESHOLDS, seed=0) is None
    assert load_report_meta(report).default_grouping is None
