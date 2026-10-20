import math

import numpy as np
import pytest

from in_tfm.density_peaks import PeakDistance, comparable_rows, density_peak_forest, forest_trees


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


def naive_distance(a: list[float], b: list[float], distance: PeakDistance) -> float:
    if distance == "weighted_jaccard":
        union = sum(max(x, y) for x, y in zip(a, b))
        return 1.0 - sum(min(x, y) for x, y in zip(a, b)) / union if union > 0 else 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def naive_kth_distance_of_each_hit(rows: list[list[float]], k: int, distance: PeakDistance) -> list[float]:
    """Over hits, not distinct points, so copies are counted by simply being there."""
    kth = []
    for i, row in enumerate(rows):
        others = sorted(naive_distance(row, other, distance) for j, other in enumerate(rows) if j != i)
        kth.append(others[min(k, len(others)) - 1])
    return kth


def naive_nearest_denser(
    rows: list[list[float]], ranks: list[int], distance: PeakDistance
) -> tuple[list[int], list[float]]:
    parents, deltas = [], []
    for i, row in enumerate(rows):
        parent, delta = -1, math.inf
        for j, other in enumerate(rows):
            if ranks[j] > ranks[i] and naive_distance(row, other, distance) < delta:
                parent, delta = j, naive_distance(row, other, distance)
        if parent == -1:
            delta = max(naive_distance(row, other, distance) for other in rows)
        parents.append(parent)
        deltas.append(delta)
    return parents, deltas


def points_with_copies(rng: np.random.Generator) -> np.ndarray:
    """Signed points, a few of them repeated, so both copy rules are exercised."""
    points = rng.normal(size=(30, 5))
    return np.vstack([points, np.repeat(points[:3], [1, 4, 7], axis=0)])


@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
@pytest.mark.parametrize("block_elements", [10, 7 * 30 * 10, 2**26])
def test_kth_distance_matches_a_naive_loop_over_hits(distance, block_elements):
    points = points_with_copies(np.random.default_rng(2))
    forest = density_peak_forest(points, k=4, distance=distance, block_elements=block_elements)
    naive = naive_kth_distance_of_each_hit(comparable_rows(points, distance).tolist(), 4, distance)
    assert np.allclose(forest.kth_distance[forest.distinct_of_hit], naive, atol=1e-5)


def test_kth_distance_with_fewer_than_k_other_hits_is_the_farthest():
    points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    forest = density_peak_forest(points, k=5, distance="euclidean")
    naive = naive_kth_distance_of_each_hit(comparable_rows(points, "euclidean").tolist(), 5, "euclidean")
    assert np.allclose(forest.kth_distance[forest.distinct_of_hit], naive)


@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
@pytest.mark.parametrize("block_elements", [10, 7 * 30 * 10, 2**26])
def test_parents_match_a_naive_loop_over_distinct_points(distance, block_elements):
    points = points_with_copies(np.random.default_rng(3))
    forest = density_peak_forest(points, k=4, distance=distance, block_elements=block_elements)
    first_hit = np.unique(forest.distinct_of_hit, return_index=True)[1]
    distinct = comparable_rows(points, distance)[first_hit].tolist()
    parents, deltas = naive_nearest_denser(distinct, forest.density_rank.tolist(), distance)
    assert forest.parent.tolist() == parents
    assert np.allclose(forest.delta, deltas, atol=1e-5)


def test_density_ranks_match_a_naive_sort():
    forest = density_peak_forest(points_with_copies(np.random.default_rng(4)), k=4)
    by_density = sorted(range(len(forest.counts)), key=lambda p: (-forest.kth_distance[p], forest.counts[p]))
    assert [int(forest.density_rank[p]) for p in by_density] == list(range(len(forest.counts)))
