import numpy as np

from in_tfm.bucketing import (
    assign_buckets,
    bucket_quotas,
    bucketed_sample,
    merge_small_buckets,
    merged_bucket_edges,
)


def test_quotas_spill_over_from_small_buckets():
    quotas = bucket_quotas(np.array([100, 2, 100, 100]), budget=30)
    assert quotas[1] == 2
    assert quotas.sum() == 30


def test_quotas_take_everything_when_budget_is_large():
    sizes = np.array([5, 0, 7])
    assert bucket_quotas(sizes, budget=1000).tolist() == sizes.tolist()


def test_max_lands_in_last_bucket():
    edges = np.linspace(0, 1, 5)
    assert assign_buckets(np.array([0.0, 1.0]), edges).tolist() == [0, 3]


def test_sample_keeps_the_tail():
    rng = np.random.default_rng(0)
    values = np.concatenate([np.full(10_000, 0.1), np.full(5, 5.0)]) + rng.random(10_005) * 0.01
    kept, buckets = bucketed_sample(values, n_buckets=10, budget=100, floor=0.0, min_bucket_size=0, rng=rng)
    assert len(kept) == 100
    assert (values[kept] > 4).sum() == 5
    assert (np.diff(kept) > 0).all()
    assert buckets[-1] == 9


def test_nothing_above_floor():
    kept, buckets = bucketed_sample(
        np.array([-1.0, 0.0]), n_buckets=4, budget=10, floor=0.0, min_bucket_size=100, rng=np.random.default_rng(0)
    )
    assert len(kept) == 0 and len(buckets) == 0


def test_small_top_buckets_merge_into_the_one_below():
    sizes = np.array([500, 300, 233, 64, 14, 8])
    assert merge_small_buckets(sizes, min_size=100).tolist() == [0, 1, 2, 2, 2, 2]


def test_small_lowest_group_joins_the_group_above():
    sizes = np.array([20, 400, 300, 10])
    assert merge_small_buckets(sizes, min_size=100).tolist() == [0, 0, 1, 1]


def test_everything_merges_when_total_is_small():
    assert merge_small_buckets(np.array([10, 5, 3]), min_size=100).tolist() == [0, 0, 0]


def test_merged_edges_drop_the_boundaries_inside_merged_buckets():
    values = np.concatenate([np.full(500, 0.1), np.full(300, 1.1), np.full(5, 1.9), np.array([4.0])])
    edges = merged_bucket_edges(values, n_buckets=4, floor=0.0, min_bucket_size=100)
    assert edges[0] == 0.0 and edges[-1] == 4.0
    assert len(edges) < 5
