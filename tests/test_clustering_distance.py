import numpy as np
from sklearn.metrics import pairwise_distances

from in_tfm.clustering import cluster_labels, contribution_shares, weighted_jaccard_distances


def test_shares_keep_signs_apart_and_sum_to_one():
    shares = contribution_shares(np.array([[3.0, -1.0, 0.0]]))
    assert np.allclose(shares, [[0.75, 0.0, 0.0, 0.0, 0.25, 0.0]])


def naive_weighted_jaccard(x: np.ndarray, y: np.ndarray) -> float:
    return 1.0 - np.minimum(x, y).sum() / np.maximum(x, y).sum()


def test_matches_a_naive_pair_by_pair_implementation():
    """Signed points, so both halves of the shares are used. Chunk 7 makes the loop run several times."""
    points = np.random.default_rng(0).normal(size=(30, 8))
    expected = pairwise_distances(contribution_shares(points), metric=naive_weighted_jaccard)
    assert np.allclose(weighted_jaccard_distances(points, chunk=7), expected)


def test_distance_matches_the_hand_worked_example():
    a = np.array([1.4, 0.9, 0.2, 0.0])
    b = np.array([1.2, 0.1, 0.3, 0.5])
    p, q = a / a.sum(), b / b.sum()
    expected = 1 - np.minimum(p, q).sum() / np.maximum(p, q).sum()
    assert np.isclose(weighted_jaccard_distances(np.stack([a, b]))[0, 1], expected)


def test_same_pattern_at_another_scale_is_distance_zero():
    a = np.array([2.0, -1.0, 0.5])
    distances = weighted_jaccard_distances(np.stack([a, 3 * a]))
    assert np.allclose(distances, 0.0)


def test_opposite_signs_on_the_same_dims_are_distance_one():
    a = np.array([1.0, 2.0, 0.0])
    assert np.isclose(weighted_jaccard_distances(np.stack([a, -a]))[0, 1], 1.0)


def test_a_dim_counts_in_proportion_to_its_size():
    base = np.array([1.0, 1.0, 0.0])
    small_extra = np.array([1.0, 1.0, 0.1])
    big_extra = np.array([1.0, 1.0, 1.0])
    distances = weighted_jaccard_distances(np.stack([base, small_extra, big_extra]))
    assert distances[0, 1] < distances[0, 2]


def test_chunks_give_the_same_matrix():
    points = np.random.default_rng(0).normal(size=(30, 8))
    assert np.allclose(weighted_jaccard_distances(points, chunk=7), weighted_jaccard_distances(points))


def test_weighted_jaccard_separates_two_groups_with_different_dims():
    rng = np.random.default_rng(0)
    group_a = np.zeros((8, 20))
    group_a[:, :5] = rng.uniform(1.0, 1.2, (8, 5))
    group_b = np.zeros((8, 20))
    group_b[:, 10:15] = rng.uniform(1.0, 1.2, (8, 5))
    labels = cluster_labels(np.vstack([group_a, group_b]), min_cluster_size=4, min_samples=2, distance="weighted_jaccard")
    assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1
    assert labels[0] != labels[8] and -1 not in labels
