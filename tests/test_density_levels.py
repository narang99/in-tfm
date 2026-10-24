import numpy as np
import pytest

from in_tfm.density_levels import (
    NOT_KEPT, DensityGraph, density_levels, elder_of_each_point, level_shares, points_kept_per_level,
)


def chain_graph(kth_distance: list[float], links: list[tuple[int, int]], counts: list[int] | None = None) -> DensityGraph:
    """Each link is listed as a neighbour of its first point only, so joins through reverse neighbours are tested too."""
    n = len(kth_distance)
    neighbours = np.full((n, 2), -1)
    filled = [0] * n
    for a, b in links:
        neighbours[a, filled[a]] = b
        filled[a] += 1
    counts = np.array(counts or [1] * n)
    return DensityGraph(
        kth_distance=np.array(kth_distance), counts=counts, neighbours=neighbours,
        distinct_of_hit=np.repeat(np.arange(n), counts),
    )


def test_level_shares_step_up_to_all_hits():
    assert level_shares(0.25) == [0.25, 0.5, 0.75, 1.0]
    assert len(level_shares(0.05)) == 20 and level_shares(0.05)[-1] == 1.0


def test_points_kept_count_copies():
    assert points_kept_per_level(np.array([3, 1, 1, 1]), [0.5, 1.0]) == [1, 4]


def test_two_dense_cores_meet_through_a_sparse_bridge_and_keep_the_elder():
    # points 0 and 1 are one core, 3 and 4 another, 2 the sparse bridge between them
    graph = chain_graph([0.1, 0.2, 0.9, 0.15, 0.3], [(1, 0), (4, 3), (2, 1), (3, 2)])
    order = np.argsort(graph.kth_distance, kind="stable")
    elders = elder_of_each_point(order, graph.neighbours, [4, 5])
    assert elders[0].tolist() == [0, 0, NOT_KEPT, 3, 3]
    assert elders[1].tolist() == [0, 0, 0, 0, 0]


def test_levels_number_groups_by_appearance_and_grey_out_small_ones():
    graph = chain_graph([0.1, 0.2, 0.9, 0.15, 0.3], [(1, 0), (4, 3), (2, 1), (3, 2)])
    probabilities = np.eye(2)[[0, 0, 1, 1, 1]]
    result = density_levels(graph, probabilities, share_step=0.2, min_group_hits=2)
    by_share = {level.share: level for level in result.levels}
    assert by_share[0.4].groups == []
    assert [g.number for g in by_share[0.6].groups] == [1]
    assert result.hit_groups[2].tolist() == [1, 1, NOT_KEPT, 0, NOT_KEPT]
    assert [(g.number, g.n_hits) for g in by_share[0.8].groups] == [(1, 2), (2, 2)]
    assert [(g.number, g.n_hits, g.clusters) for g in by_share[1.0].groups] == [(1, 5, [(1, 3), (0, 2)])]


@pytest.mark.parametrize("step", [0.05, 0.3])
def test_every_hit_is_kept_at_the_last_level(step):
    rng = np.random.default_rng(0)
    n = 40
    graph = DensityGraph(
        kth_distance=rng.random(n), counts=np.ones(n, dtype=int), neighbours=rng.integers(0, n, (n, 3)),
        distinct_of_hit=np.arange(n),
    )
    result = density_levels(graph, np.ones((n, 1)), step, min_group_hits=1)
    assert (result.hit_groups[-1] != NOT_KEPT).all()
    assert [level.n_hits for level in result.levels] == sorted(level.n_hits for level in result.levels)
