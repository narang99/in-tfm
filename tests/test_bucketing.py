import numpy as np

from in_tfm.hit_selection.strategies.bucketed import (
    bucket_quotas,
    bucketed_sample,
    merge_small_buckets,
    plan_buckets,
    raw_bucket_of_each_value,
)


def test_quotas_spill_over_from_small_buckets():
    quotas = bucket_quotas([100, 2, 100, 100], budget=30)
    assert quotas[1] == 2
    assert sum(quotas) == 30


def test_quotas_take_everything_when_budget_is_large():
    assert bucket_quotas([5, 0, 7], budget=1000) == [5, 0, 7]


def test_max_lands_in_last_bucket():
    edges = np.linspace(0, 1, 5).tolist()
    assert raw_bucket_of_each_value(np.array([0.0, 1.0]), edges).tolist() == [0, 3]


def test_sample_keeps_the_tail():
    rng = np.random.default_rng(0)
    values = np.concatenate([np.full(10_000, 0.1), np.full(5, 5.0)]) + rng.random(10_005) * 0.01
    sample = bucketed_sample(values, n_buckets=10, budget=100, floor=0.0, min_bucket_size=0, rng=rng)
    assert len(sample.kept_value_idx) == 100
    assert (values[sample.kept_value_idx] > 4).sum() == 5
    assert (np.diff(sample.kept_value_idx) > 0).all()
    assert sample.group_of_kept[-1] == 9


def test_nothing_above_floor():
    sample = bucketed_sample(
        np.array([-1.0, 0.0]), n_buckets=4, budget=10, floor=0.0, min_bucket_size=100, rng=np.random.default_rng(0)
    )
    assert len(sample.kept_value_idx) == 0 and sample.layout is None


def test_small_top_buckets_merge_into_the_one_below():
    assert merge_small_buckets([500, 300, 233, 64, 14, 8], min_size=100) == [[0], [1], [2, 3, 4, 5]]


def test_small_lowest_group_joins_the_group_above():
    assert merge_small_buckets([20, 400, 300, 10], min_size=100) == [[0, 1], [2, 3]]


def test_everything_merges_when_total_is_small():
    assert merge_small_buckets([10, 5, 3], min_size=100) == [[0, 1, 2]]


def test_merged_edges_drop_the_boundaries_inside_merged_buckets():
    values = np.concatenate([np.full(500, 0.1), np.full(300, 1.1), np.full(5, 1.9), np.array([4.0])])
    edges = plan_buckets(values, n_buckets=4, floor=0.0, min_bucket_size=100).group_edges()
    assert edges[0] == 0.0 and edges[-1] == 4.0
    assert len(edges) < 5
