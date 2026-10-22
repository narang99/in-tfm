import numpy as np
import pytest

from in_tfm.density_peaks import DensityPeakForest, ForestTrees, RootCuts, calculate_probs, density_peak_forest, forest_trees
from in_tfm.density_peaks.probabilities import geodesic_distances, knn_graph, peak_cores, softmax_over_trees


def line_forest() -> tuple[DensityPeakForest, ForestTrees]:
    """Points on a line at 0, 1, 2, 3 and 10, each joined to its one nearest. Two trees: {0, 1, 2} and {3, 4}.

    The roots are 0 and 3. Point 1 is as dense as root 0, point 2 half as dense."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    neighbours = np.array([[1], [0], [1], [2], [3]])
    forest = DensityPeakForest(
        distinct_of_hit=np.arange(5), counts=np.ones(5, dtype=int), kth_distance=np.array([1.0, 1.0, 2.0, 1.0, 7.0]),
        density_rank=np.array([4, 3, 1, 2, 0]), parent=np.array([-1, 0, 1, -1, 3]), delta=np.zeros(5),
        neighbours=neighbours, neighbour_distance=np.abs(x[:, None] - x[neighbours])[np.arange(5), 0][:, None],
    )
    trees = ForestTrees(cuts=RootCuts(rho=0.0, delta=0.0), roots=np.array([0, 3]), tree=np.array([0, 0, 0, 1, 1]),
                        depth=np.array([0, 1, 2, 0, 1]))
    return forest, trees


def test_the_peak_keeps_points_within_its_share_of_the_root_density():
    forest, trees = line_forest()
    assert [c.tolist() for c in peak_cores(forest, trees, 0.5)] == [[0, 1, 2], [3]]
    assert [c.tolist() for c in peak_cores(forest, trees, 1.0)] == [[0, 1], [3]]


def test_geodesic_distance_sums_the_hops_to_the_closest_peak_point():
    forest, trees = line_forest()
    distances = geodesic_distances(knn_graph(forest), peak_cores(forest, trees, 1.0))
    # point 4 reaches T0's peak {0, 1} only through 3 and 2
    assert np.allclose(distances[:, 0], [0, 0, 1, 2, 9])
    assert np.allclose(distances[:, 1], [3, 2, 1, 0, 7])


def test_softmax_rows_sum_to_one_and_favour_the_closer_tree():
    probabilities = softmax_over_trees(np.array([[0.1, 0.3], [5.0, 5.0], [1000.0, 1001.0], [2.0, np.inf]]), tau=0.1)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert np.allclose(probabilities[1], [0.5, 0.5])
    # far rows still get a probability rather than underflowing to nothing
    assert probabilities[2, 0] > 0.99
    assert np.allclose(probabilities[3], [1, 0])


def test_two_separate_blobs_each_keep_their_own_tree():
    rng = np.random.default_rng(0)
    blob_a = np.zeros((20, 6))
    blob_a[:, :2] = rng.uniform(1.0, 1.2, (20, 2))
    blob_b = np.zeros((20, 6))
    blob_b[:, 4:] = rng.uniform(1.0, 1.2, (20, 2))
    forest = density_peak_forest(np.vstack([blob_a, blob_b]), k=4)
    trees = forest_trees(forest, RootCuts(rho=float("-inf"), delta=0.5))
    probabilities = calculate_probs(forest, trees, peak_share=0.8)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(probabilities.argmax(axis=1), trees.tree)


@pytest.mark.parametrize("peak_share", [0.5, 0.8, 1.0])
def test_a_point_out_of_reach_of_every_peak_keeps_its_own_tree(peak_share):
    forest, trees = line_forest()
    # cut every edge, so no point reaches any peak but its own
    isolated = forest.model_copy(update={"neighbour_distance": np.full((5, 1), np.inf)})
    probabilities = calculate_probs(isolated, trees, peak_share)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(probabilities.argmax(axis=1), trees.tree)
