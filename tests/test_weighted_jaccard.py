import numpy as np
import pytest

from in_tfm.weighted_jaccard import naive, vectorized


def signed_points_with_a_zero_row(rng: np.random.Generator) -> np.ndarray:
    """Signed, so both halves of the shares are used. The zero row checks the empty-union case."""
    return np.vstack([rng.normal(size=(30, 8)), np.zeros((1, 8))])


def test_shares_match_the_naive_loop():
    points = signed_points_with_a_zero_row(np.random.default_rng(0))
    assert np.allclose(vectorized.contribution_shares(points), naive.contribution_shares(points))


def test_block_matches_the_naive_loop():
    shares = naive.contribution_shares(signed_points_with_a_zero_row(np.random.default_rng(1)))
    block = shares[5:12]
    assert np.allclose(vectorized.weighted_jaccard_block(block, shares), naive.weighted_jaccard_block(block, shares))


@pytest.mark.parametrize("chunk", [1, 7, 256])
def test_distances_match_the_naive_loop(chunk):
    points = signed_points_with_a_zero_row(np.random.default_rng(2))
    expected = naive.weighted_jaccard_distances(points)
    assert np.allclose(vectorized.weighted_jaccard_distances(points, chunk=chunk), expected)


def test_shares_keep_signs_apart_and_sum_to_one():
    shares = vectorized.contribution_shares(np.array([[3.0, -1.0, 0.0]]))
    assert np.allclose(shares, [[0.75, 0.0, 0.0, 0.0, 0.25, 0.0]])


def test_an_all_zero_row_stays_all_zero():
    assert np.all(vectorized.contribution_shares(np.zeros((1, 3))) == 0.0)


def test_distance_matches_the_hand_worked_example():
    a = np.array([1.4, 0.9, 0.2, 0.0])
    b = np.array([1.2, 0.1, 0.3, 0.5])
    p, q = a / a.sum(), b / b.sum()
    expected = 1 - np.minimum(p, q).sum() / np.maximum(p, q).sum()
    assert np.isclose(vectorized.weighted_jaccard_distances(np.stack([a, b]))[0, 1], expected)


def test_same_pattern_at_another_scale_is_distance_zero():
    a = np.array([2.0, -1.0, 0.5])
    assert np.allclose(vectorized.weighted_jaccard_distances(np.stack([a, 3 * a])), 0.0)


def test_opposite_signs_on_the_same_dims_are_distance_one():
    a = np.array([1.0, 2.0, 0.0])
    assert np.isclose(vectorized.weighted_jaccard_distances(np.stack([a, -a]))[0, 1], 1.0)


def test_a_dim_counts_in_proportion_to_its_size():
    base = np.array([1.0, 1.0, 0.0])
    small_extra = np.array([1.0, 1.0, 0.1])
    big_extra = np.array([1.0, 1.0, 1.0])
    distances = vectorized.weighted_jaccard_distances(np.stack([base, small_extra, big_extra]))
    assert distances[0, 1] < distances[0, 2]
