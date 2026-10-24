"""The torch forest against the numpy one, on torch's CPU device, so the check runs without a GPU."""

import numpy as np
import pytest
import torch

from in_tfm.density_peaks import torch_forest, vectorized
from in_tfm.weighted_jaccard import weighted_jaccard_distances

DISTANCES = ["euclidean", "weighted_jaccard"]
BLOCK_ELEMENTS = [10, 7 * 33, 2**27]
"""One row per block, a few rows per block, and one block."""


def points_with_copies(rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(30, 5))
    return np.vstack([points, np.repeat(points[:3], [1, 4, 7], axis=0)])


def assert_same_forest(forest, expected) -> None:
    assert np.array_equal(forest.distinct_of_hit, expected.distinct_of_hit)
    assert np.array_equal(forest.counts, expected.counts)
    assert np.allclose(forest.kth_distance, expected.kth_distance)
    assert np.array_equal(forest.density_rank, expected.density_rank)
    assert np.array_equal(forest.parent, expected.parent)
    assert np.allclose(forest.delta, expected.delta)
    assert np.array_equal(forest.neighbours, expected.neighbours)
    assert np.allclose(forest.neighbour_distance, expected.neighbour_distance)


@pytest.mark.parametrize("block_elements", BLOCK_ELEMENTS)
@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("distance", DISTANCES)
def test_torch_forest_matches_numpy(distance, normalise, block_elements):
    points = points_with_copies(np.random.default_rng(6))
    expected = vectorized.density_peak_forest(points, k=4, distance=distance, normalise=normalise)
    forest = torch_forest.density_peak_forest(
        points, k=4, distance=distance, normalise=normalise, device="cpu", block_elements=block_elements,
    )
    assert_same_forest(forest, expected)


def test_torch_forest_with_fewer_points_than_k_matches_numpy():
    points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    expected = vectorized.density_peak_forest(points, k=5)
    assert_same_forest(torch_forest.density_peak_forest(points, k=5, device="cpu"), expected)


def test_torch_forest_on_points_tied_at_zero_distance_matches_numpy():
    """Many points share a pattern at different scales, so they tie at distance 0 once normalised away."""
    rng = np.random.default_rng(9)
    base = rng.normal(size=(6, 4))
    points = np.vstack([base * scale for scale in (1.0, 2.0, 3.0)] + [rng.normal(size=(10, 4))])
    expected = vectorized.density_peak_forest(points, k=3, normalise=False)
    assert_same_forest(torch_forest.density_peak_forest(points, k=3, normalise=False, device="cpu"), expected)


@pytest.mark.parametrize("normalise", [True, False])
def test_pairwise_distances_match_numpy(normalise):
    points = np.random.default_rng(3).normal(size=(25, 6))
    expected = weighted_jaccard_distances(points, normalise=normalise)
    result = torch_forest.pairwise_distances(points, "weighted_jaccard", normalise, "cpu", block_elements=7 * 25)
    assert np.allclose(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_forest_matches_numpy():
    points = np.random.default_rng(11).normal(size=(300, 16))
    expected = vectorized.density_peak_forest(points, k=5)
    assert_same_forest(torch_forest.density_peak_forest(points, k=5, device="cuda"), expected)
