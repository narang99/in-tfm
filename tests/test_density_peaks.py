import numpy as np
import pytest

from in_tfm.density_peaks import comparable_rows, density_peak_forest, forest_trees, merge_copies, naive, vectorized


def two_blobs(rng: np.random.Generator) -> np.ndarray:
    blob_a = np.zeros((12, 10))
    blob_a[:, :3] = rng.uniform(1.0, 1.2, (12, 3))
    blob_b = np.zeros((8, 10))
    blob_b[:, 6:9] = rng.uniform(1.0, 1.2, (8, 3))
    return np.vstack([blob_a, blob_b])


def test_copies_are_merged_and_counted():
    point = np.array([1.0, 2.0, 0.0])
    forest = density_peak_forest(np.stack([point, point, point, [0.0, 0.0, 1.0]]), k=2)
    assert sorted(forest.counts.tolist()) == [1, 3]
    assert forest.distinct_of_hit[0] == forest.distinct_of_hit[1] == forest.distinct_of_hit[2]


def test_a_point_seen_more_than_k_times_has_kth_distance_zero():
    forest = density_peak_forest(np.vstack([np.tile([1.0, 0.0, 0.0], (6, 1)), [[0.0, 1.0, 0.0]]]), k=5)
    repeated = forest.distinct_of_hit[0]
    assert forest.kth_distance[repeated] == 0.0


def test_kth_distance_counts_neighbour_copies():
    """The lone point's 2 nearest, counted with copies, are both copies of the near point."""
    near, far = np.array([1.0, 0.1, 0.0]), np.array([0.0, 0.0, 1.0])
    lone = np.array([1.0, 0.3, 0.0])
    points = np.vstack([lone, near, near, far])
    forest = density_peak_forest(points, k=2, distance="euclidean")
    unit = lambda v: v / np.linalg.norm(v)
    assert np.isclose(forest.kth_distance[forest.distinct_of_hit[0]], np.linalg.norm(unit(lone) - unit(near)))


def test_every_point_but_the_densest_links_to_a_denser_one():
    forest = density_peak_forest(two_blobs(np.random.default_rng(0)), k=3)
    linked = forest.parent >= 0
    assert (~linked).sum() == 1
    assert np.all(forest.density_rank[forest.parent[linked]] > forest.density_rank[linked])


def test_two_blobs_give_two_trees_and_no_noise():
    points = two_blobs(np.random.default_rng(0))
    forest = density_peak_forest(points, k=3, distance="weighted_jaccard")
    labels = forest_trees(forest, root_delta=0.5, root_density_quantile=0.5).tree[forest.distinct_of_hit]
    assert len(set(labels[:12])) == 1 and len(set(labels[12:])) == 1
    assert labels[0] != labels[12] and labels.min() == 0


def test_trees_are_numbered_largest_first_and_roots_have_depth_zero():
    forest = density_peak_forest(two_blobs(np.random.default_rng(0)), k=3)
    trees = forest_trees(forest, root_delta=0.5, root_density_quantile=0.5)
    hits_per_tree = np.bincount(trees.tree[forest.distinct_of_hit])
    assert hits_per_tree.tolist() == [12, 8]
    assert np.all(trees.depth[trees.roots] == 0)
    assert np.all(trees.tree[trees.roots] == np.arange(len(trees.roots)))


def test_small_blocks_give_the_same_forest():
    points = np.random.default_rng(1).normal(size=(40, 6))
    whole = density_peak_forest(points, k=4)
    blocked = density_peak_forest(points, k=4, block_elements=7 * 40 * 12)
    assert np.allclose(whole.kth_distance, blocked.kth_distance)
    assert np.array_equal(whole.parent, blocked.parent)


def points_with_copies(rng: np.random.Generator) -> np.ndarray:
    """Signed points, a few of them repeated, so both copy rules are exercised."""
    points = rng.normal(size=(30, 5))
    return np.vstack([points, np.repeat(points[:3], [1, 4, 7], axis=0)])


def merged(points: np.ndarray, distance) -> tuple[np.ndarray, np.ndarray]:
    distinct, _, counts = merge_copies(comparable_rows(points, distance))
    return distinct, counts


DISTANCES = ["euclidean", "weighted_jaccard"]
BLOCK_ELEMENTS = [10, 7 * 30 * 10, 2**26]
"""One row per block, a few rows per block, and one block."""


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("block_elements", BLOCK_ELEMENTS)
def test_kth_distances_match_naive(distance, block_elements):
    rows, counts = merged(points_with_copies(np.random.default_rng(2)), distance)
    expected = naive.kth_distances(rows, counts, 4, distance)
    assert np.allclose(vectorized.kth_distances(rows, counts, 4, distance, block_elements), expected)


@pytest.mark.parametrize("distance", DISTANCES)
def test_kth_distances_with_fewer_than_k_other_hits_match_naive(distance):
    rows, counts = merged(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), distance)
    expected = naive.kth_distances(rows, counts, 5, distance)
    assert np.allclose(vectorized.kth_distances(rows, counts, 5, distance, 2**26), expected)


def test_density_ranks_match_naive():
    rows, counts = merged(points_with_copies(np.random.default_rng(4)), "weighted_jaccard")
    kth = vectorized.kth_distances(rows, counts, 4, "weighted_jaccard", 2**26)
    assert np.array_equal(vectorized.density_ranks(kth, counts), naive.density_ranks(kth, counts))


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("block_elements", BLOCK_ELEMENTS)
def test_nearest_denser_matches_naive(distance, block_elements):
    rows, counts = merged(points_with_copies(np.random.default_rng(3)), distance)
    ranks = vectorized.density_ranks(vectorized.kth_distances(rows, counts, 4, distance, 2**26), counts)
    parent, delta = vectorized.nearest_denser(rows, ranks, distance, block_elements)
    naive_parent, naive_delta = naive.nearest_denser(rows, ranks, distance)
    assert np.array_equal(parent, naive_parent)
    assert np.allclose(delta, naive_delta)


@pytest.mark.parametrize("root_delta", [0.05, 0.3, 2.0])
def test_forest_trees_match_naive(root_delta):
    forest = density_peak_forest(points_with_copies(np.random.default_rng(5)), k=4)
    trees = vectorized.forest_trees(forest, root_delta, 0.5)
    naive_trees = naive.forest_trees(forest, root_delta, 0.5)
    assert np.array_equal(trees.roots, naive_trees.roots)
    assert np.array_equal(trees.tree, naive_trees.tree)
    assert np.array_equal(trees.depth, naive_trees.depth)


@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("distance", DISTANCES)
def test_whole_forest_matches_naive(distance, normalise):
    points = points_with_copies(np.random.default_rng(6))
    forest = vectorized.density_peak_forest(points, k=4, distance=distance, normalise=normalise, block_elements=10)
    naive_forest = naive.density_peak_forest(points, k=4, distance=distance, normalise=normalise)
    assert np.array_equal(forest.distinct_of_hit, naive_forest.distinct_of_hit)
    assert np.array_equal(forest.counts, naive_forest.counts)
    assert np.allclose(forest.kth_distance, naive_forest.kth_distance)
    assert np.array_equal(forest.density_rank, naive_forest.density_rank)
    assert np.array_equal(forest.parent, naive_forest.parent)
    assert np.allclose(forest.delta, naive_forest.delta)


@pytest.mark.parametrize("distance", DISTANCES)
def test_without_normalising_a_scaled_copy_is_its_own_point(distance):
    point = np.array([1.0, -2.0, 0.5])
    assert len(density_peak_forest(np.stack([point, 3 * point]), k=1, distance=distance).counts) == 1
    assert len(density_peak_forest(np.stack([point, 3 * point]), k=1, distance=distance, normalise=False).counts) == 2
