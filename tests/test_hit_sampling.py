import numpy as np

from in_tfm.neuron_report import sampled_hit_indices

LABELS = np.array([0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 1])


def test_the_same_hits_are_picked_every_time():
    first = sampled_hit_indices(LABELS, 1, max_n=3)
    assert np.array_equal(first, sampled_hit_indices(LABELS, 1, max_n=3))


def test_only_members_of_the_cluster_are_picked_and_at_most_max_n():
    picked = sampled_hit_indices(LABELS, 1, max_n=4)
    assert len(picked) == 4
    assert set(picked) <= set(np.flatnonzero(LABELS == 1))


def test_a_small_cluster_is_shown_whole():
    assert sorted(sampled_hit_indices(LABELS, 2, max_n=5)) == [7]


def test_clusters_are_not_all_ordered_the_same_way():
    labels = np.array([0] * 20 + [1] * 20)
    a = sampled_hit_indices(labels, 0, max_n=20) % 20
    b = sampled_hit_indices(labels, 1, max_n=20) % 20
    assert not np.array_equal(a, b)
