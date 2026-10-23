import numpy as np

from in_tfm.hit_buckets import all_buckets, cluster_buckets, clusters_needed, set_buckets


def rows(*probabilities: list[float]) -> np.ndarray:
    return np.array(probabilities)


def test_a_hit_needs_as_many_clusters_as_it_takes_to_reach_the_mass():
    probabilities = rows([0.9, 0.05, 0.05, 0.0], [0.5, 0.4, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1])
    assert clusters_needed(probabilities).tolist() == [1, 2, 3]


def test_a_set_is_the_same_whichever_cluster_leads():
    probabilities = rows([0.5, 0.4, 0.1, 0.0, 0.0, 0.0], [0.4, 0.5, 0.1, 0.0, 0.0, 0.0])
    [bucket] = set_buckets(probabilities)
    assert bucket.clusters == (0, 1)
    assert bucket.hits == [0, 1]


def test_a_hit_spread_over_more_than_half_the_clusters_is_in_no_set():
    spread = rows([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    assert set_buckets(spread) == []


def test_a_sure_hit_is_in_every_cluster_tier_it_reaches_and_no_set():
    sure = rows([0.95, 0.05, 0.0, 0.0])
    assert [b.name for b in cluster_buckets(sure)] == ["cluster 0 >= 0.9", "cluster 0 >= 0.8", "cluster 0 >= 0.7"]
    assert set_buckets(sure) == []


def test_buckets_come_largest_first_and_empty_ones_are_absent():
    probabilities = rows([0.95, 0.05, 0.0, 0.0], [0.96, 0.04, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1])
    buckets = all_buckets(probabilities)
    assert buckets[0].name == "cluster 0 >= 0.9"
    assert all(bucket.hits for bucket in buckets)
