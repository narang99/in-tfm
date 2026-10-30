"""The IVF forest against the numpy one, on torch's CPU device.

- These points have no copies, so merging changes nothing and the two forests build from the same rows.
- They also have no near ties: float32 can swap two neighbours within about 1e-6, as in `torch_forest`.
- With every list probed, every pair is compared, so the forests agree exactly.
- With few lists probed, the forest is approximate, and only its shape is checked.
"""

import numpy as np
import pytest
import torch

from in_tfm.density_peaks import comparable_rows, forest_trees, ivf_forest, merge_copies, vectorized
from in_tfm.density_peaks.ivf_forest import IvfSettings, ivf_density_peak_forest

FLOAT32_TOLERANCE = 1e-5


def clustered_points(rng: np.random.Generator, n: int = 300) -> np.ndarray:
    centres = rng.normal(scale=4.0, size=(4, 6))
    return centres[rng.integers(0, 4, size=n)] + rng.normal(size=(n, 6))


def in_merge_order(points: np.ndarray, distance: str, normalise: bool) -> np.ndarray:
    """`merge_copies` sorts the distinct rows, and density ties go to the lower index.

    - Mutual kth neighbours tie on kth distance, so the order decides some ranks and so some parents.
    - Giving the points already sorted makes both forests number them the same way."""
    _, distinct_of_hit, _ = merge_copies(comparable_rows(points, distance, normalise))
    order = np.empty_like(distinct_of_hit)
    order[distinct_of_hit] = np.arange(len(points))
    return points[order]


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(ivf_forest, "BLOCK_ELEMENTS", 500)
    monkeypatch.setattr(ivf_forest, "PROBE_PAIRS", 70)


@pytest.mark.parametrize("normalise", [True, False])
@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
@pytest.mark.parametrize("list_length", [4, 32])
@pytest.mark.parametrize("exact_on_device", [False, True])
def test_probing_every_list_matches_numpy(distance, normalise, list_length, exact_on_device):
    points = in_merge_order(clustered_points(np.random.default_rng(0), 200), distance, normalise)
    expected = vectorized.density_peak_forest(points, k=4, distance=distance, normalise=normalise)
    assert np.array_equal(expected.distinct_of_hit, np.arange(len(points)))
    settings = IvfSettings(n_lists=9, n_probe=9, peak_probe=9, n_list_neighbours=list_length,
                           exact_on_device=exact_on_device)
    forest, _ = ivf_density_peak_forest(points, settings, k=4, distance=distance, normalise=normalise, device="cpu")
    assert np.allclose(forest.kth_distance, expected.kth_distance, atol=FLOAT32_TOLERANCE)
    assert np.array_equal(forest.density_rank, expected.density_rank)
    assert np.array_equal(forest.parent, expected.parent)
    assert np.allclose(forest.delta, expected.delta, atol=FLOAT32_TOLERANCE)
    assert np.array_equal(forest.neighbours, expected.neighbours)
    assert np.allclose(forest.neighbour_distance, expected.neighbour_distance, atol=FLOAT32_TOLERANCE)


def test_a_short_list_leaves_more_local_peaks():
    points = clustered_points(np.random.default_rng(4))
    short = IvfSettings(n_lists=9, n_probe=9, peak_probe=9, n_list_neighbours=4)
    long = short.model_copy(update={"n_list_neighbours": 32})
    _, short_stats = ivf_density_peak_forest(points, short, k=4, device="cpu")
    _, long_stats = ivf_density_peak_forest(points, long, k=4, device="cpu")
    assert long_stats.local_peaks < short_stats.local_peaks


@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
def test_probing_every_list_finds_what_the_exhaustive_search_finds(distance):
    rows = ivf_forest.Rows(clustered_points(np.random.default_rng(7)), distance, False, "cpu")
    lists = ivf_forest.build_lists(rows, 9, seed=0)
    queries = torch.arange(len(rows))
    probed = ivf_forest.probed_search(rows, lists, queries, 9, 6)
    exhaustive = ivf_forest.exhaustive_search(rows, queries, 6, None)
    assert torch.equal(probed[0], exhaustive[0])
    assert torch.allclose(probed[1], exhaustive[1])


def test_a_partial_probe_still_builds_one_forest_with_copies():
    rng = np.random.default_rng(5)
    points = np.vstack([clustered_points(rng), np.repeat(clustered_points(rng, 2), 20, axis=0)])
    settings = IvfSettings(n_lists=16, n_probe=2, peak_probe=4, n_list_neighbours=8)
    forest, _ = ivf_density_peak_forest(points, settings, k=4, normalise=False, device="cpu")
    assert (forest.parent == -1).sum() == 1
    assert np.array_equal(np.sort(forest.density_rank), np.arange(len(points)))
    denser = forest.parent >= 0
    assert (forest.density_rank[forest.parent[denser]] > forest.density_rank[denser]).all()
    assert len(forest_trees(forest).roots) >= 1


@pytest.mark.parametrize("distance", ["euclidean", "weighted_jaccard"])
def test_cell_search_over_every_list_finds_what_the_exhaustive_search_finds(distance):
    rows = ivf_forest.Rows(clustered_points(np.random.default_rng(8)), distance, False, "cpu")
    lists = ivf_forest.build_lists(rows, 9, seed=0)
    queries = torch.arange(len(rows))
    by_cell = ivf_forest.cell_search(rows, lists, queries, 9, 6)
    exhaustive = ivf_forest.exhaustive_search(rows, queries, 6, None)
    assert torch.equal(by_cell[0], exhaustive[0])
    assert torch.allclose(by_cell[1], exhaustive[1])


def test_cell_search_respects_the_allowed_pairs():
    rows = ivf_forest.Rows(clustered_points(np.random.default_rng(9)), "weighted_jaccard", False, "cpu")
    lists = ivf_forest.build_lists(rows, 9, seed=0)
    queries = torch.arange(0, len(rows), 7)
    only_higher: ivf_forest.AllowedPairs = lambda q, m: m[None, :] > q[:, None]  # noqa: E731
    by_cell = ivf_forest.cell_search(rows, lists, queries, 9, 3, only_higher)
    exhaustive = ivf_forest.exhaustive_search(rows, queries, 3, only_higher)
    assert torch.equal(by_cell[0], exhaustive[0])


def test_a_partial_cell_probe_still_builds_one_forest():
    points = clustered_points(np.random.default_rng(10), 400)
    settings = IvfSettings(n_lists=16, n_probe=3, peak_probe=5, n_list_neighbours=8, probe_by="cell")
    forest, _ = ivf_density_peak_forest(points, settings, k=4, normalise=False, device="cpu")
    assert (forest.parent == -1).sum() == 1
    denser = forest.parent >= 0
    assert (forest.density_rank[forest.parent[denser]] > forest.density_rank[denser]).all()


def test_the_union_of_probes_finds_at_least_what_each_point_finds():
    rows = ivf_forest.Rows(clustered_points(np.random.default_rng(11), 500), "weighted_jaccard", False, "cpu")
    lists = ivf_forest.build_lists(rows, 16, seed=0)
    queries = torch.arange(len(rows))
    by_union = ivf_forest.cell_search(rows, lists, queries, 3, 6, union=True)
    by_point = ivf_forest.probed_search(rows, lists, queries, 3, 6)
    assert (by_union[1] <= by_point[1] + FLOAT32_TOLERANCE).all()


def test_copies_of_rows_leave_no_empty_lists_and_every_point_its_neighbours():
    rng = np.random.default_rng(13)
    points = np.repeat(clustered_points(rng, 30), 40, axis=0)
    rows = ivf_forest.Rows(points, "weighted_jaccard", False, "cpu")
    lists = ivf_forest.build_lists(rows, 60, seed=0)
    assert (lists.starts[1:] > lists.starts[:-1]).all()
    settings = IvfSettings(n_lists=60, n_probe=4, peak_probe=8, n_list_neighbours=8)
    forest, _ = ivf_density_peak_forest(points, settings, k=4, normalise=False, device="cpu")
    assert (forest.neighbours >= 0).all()
