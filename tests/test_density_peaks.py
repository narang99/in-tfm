import numpy as np

from in_tfm import density_peaks
from in_tfm.density_peaks import density_peak_forest, forest_trees


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
    assert forest.point_of_hit[0] == forest.point_of_hit[1] == forest.point_of_hit[2]


def test_a_point_seen_more_than_k_times_has_kth_distance_zero():
    forest = density_peak_forest(np.vstack([np.tile([1.0, 0.0, 0.0], (6, 1)), [[0.0, 1.0, 0.0]]]), k=5)
    repeated = forest.point_of_hit[0]
    assert forest.kth_distance[repeated] == 0.0


def test_kth_distance_counts_neighbour_copies():
    """The lone point's 2 nearest, counted with copies, are both copies of the near point."""
    near, far = np.array([1.0, 0.1, 0.0]), np.array([0.0, 0.0, 1.0])
    lone = np.array([1.0, 0.3, 0.0])
    points = np.vstack([lone, near, near, far])
    forest = density_peak_forest(points, k=2, distance="euclidean")
    unit = lambda v: v / np.linalg.norm(v)
    assert np.isclose(forest.kth_distance[forest.point_of_hit[0]], np.linalg.norm(unit(lone) - unit(near)))


def test_every_point_but_the_densest_links_to_a_denser_one():
    forest = density_peak_forest(two_blobs(np.random.default_rng(0)), k=3)
    linked = forest.parent >= 0
    assert (~linked).sum() == 1
    assert np.all(forest.density_rank[forest.parent[linked]] > forest.density_rank[linked])


def test_two_blobs_give_two_trees_and_no_noise():
    points = two_blobs(np.random.default_rng(0))
    forest = density_peak_forest(points, k=3, distance="weighted_jaccard")
    labels = forest_trees(forest, root_delta=0.5, root_density_quantile=0.5).tree[forest.point_of_hit]
    assert len(set(labels[:12])) == 1 and len(set(labels[12:])) == 1
    assert labels[0] != labels[12] and labels.min() == 0


def test_trees_are_numbered_largest_first_and_roots_have_depth_zero():
    forest = density_peak_forest(two_blobs(np.random.default_rng(0)), k=3)
    trees = forest_trees(forest, root_delta=0.5, root_density_quantile=0.5)
    hits_per_tree = np.bincount(trees.tree[forest.point_of_hit])
    assert hits_per_tree.tolist() == [12, 8]
    assert np.all(trees.depth[trees.roots] == 0)
    assert np.all(trees.tree[trees.roots] == np.arange(len(trees.roots)))


def test_small_blocks_give_the_same_forest(monkeypatch):
    points = np.random.default_rng(1).normal(size=(40, 6))
    whole = density_peak_forest(points, k=4)
    monkeypatch.setattr(density_peaks, "DISTANCE_BLOCK_ELEMENTS", 7 * 40 * 12)
    blocked = density_peak_forest(points, k=4)
    assert np.allclose(whole.kth_distance, blocked.kth_distance)
    assert np.array_equal(whole.parent, blocked.parent)
