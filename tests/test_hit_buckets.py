import numpy as np

from in_tfm.hit_buckets import (
    Bucket, all_buckets, buckets_of_cluster, cluster_buckets, clusters_needed, paged_buckets, set_buckets,
)
from in_tfm.run_config import BucketsConfig

CONFIG = BucketsConfig()


def rows(*probabilities: list[float]) -> np.ndarray:
    return np.array(probabilities)


def test_a_hit_needs_as_many_clusters_as_it_takes_to_reach_the_mass():
    probabilities = rows([0.9, 0.05, 0.05, 0.0], [0.5, 0.4, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1])
    assert clusters_needed(probabilities, 0.85).tolist() == [1, 2, 3]


def test_a_set_is_the_same_whichever_cluster_leads():
    probabilities = rows([0.5, 0.4, 0.1, 0.0, 0.0, 0.0], [0.4, 0.5, 0.1, 0.0, 0.0, 0.0])
    [bucket] = set_buckets(probabilities, 0.85)
    assert bucket.clusters == (0, 1)
    assert bucket.hits == [0, 1]


def test_a_hit_spread_over_more_than_half_the_clusters_is_in_no_set():
    spread = rows([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    assert set_buckets(spread, 0.85) == []


def test_a_sure_hit_is_in_every_cluster_tier_it_reaches_and_no_set():
    sure = rows([0.95, 0.05, 0.0, 0.0])
    assert [b.name for b in cluster_buckets(sure, [0.9, 0.8, 0.7])] == [
        "cluster 0 >= 0.9", "cluster 0 >= 0.8", "cluster 0 >= 0.7",
    ]
    assert set_buckets(sure, 0.85) == []


def test_buckets_come_largest_first_and_empty_ones_are_absent():
    probabilities = rows([0.95, 0.05, 0.0, 0.0], [0.96, 0.04, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1])
    buckets = all_buckets(probabilities, CONFIG)
    assert buckets[0].name == "cluster 0 >= 0.9"
    assert all(bucket.hits for bucket in buckets)


def test_a_slug_is_a_file_safe_name_and_tiers_stay_distinct():
    def slug(name: str) -> str:
        return Bucket(name=name, clusters=(0,), hits=[0]).slug

    assert slug("clusters 0 + 4") == "clusters_0_4"
    assert slug("cluster 0 >= 0.9") != slug("cluster 0 >= 0.8")


def test_only_big_enough_buckets_are_paged_and_a_cluster_finds_its_buckets():
    small, big = Bucket(name="a", clusters=(0, 2), hits=[1]), Bucket(name="b", clusters=(1, 2), hits=list(range(10)))
    assert paged_buckets([small, big], CONFIG) == [big]
    assert buckets_of_cluster([small, big], 2) == [small, big]
    assert buckets_of_cluster([small, big], 0) == [small]
