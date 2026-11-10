import base64
import json
import re

import numpy as np

from in_tfm.persistence_report import (
    ROUTE_STOPS,
    TILE_LEVELS,
    HitDisplay,
    encoded_tile,
    persistence_data,
    points_at_capped_density,
    smallest_step,
    step_ends,
    thinned_route,
    write_persistence_page,
)
from in_tfm.presenters import HitLabel
from in_tfm.report_page import ForestData


def two_hills() -> ForestData:
    """Two roots at points 0 and 4, with a smaller bump at point 2 that is no tree's root."""
    n = 5
    return ForestData(
        rho=np.array([5.0, 1.0, 3.0, 2.5, 4.0]),
        delta=np.ones(n),
        parent_in_tree=np.array([-1, 0, 4, 4, -1]),
        tree=np.array([0, 0, 1, 1, 1]),
        depth=np.array([0, 1, 1, 1, 0]),
        counts=np.array([2, 1, 1, 1, 1]),
        roots=np.array([0, 4]),
        cut_rho=3.5,
        cut_delta=0.5,
        probabilities=np.ones((n, 2)) / 2,
        neighbours=np.array([[1], [2], [3], [4], [3]]),
        distinct_of_hit=np.array([0, 0, 1, 2, 3, 4]),
    )


def label_of(point: int) -> HitLabel:
    return HitLabel(before="</script><b>", token=f"p{point}")


def display_of(hadamards: np.ndarray | None = None) -> HitDisplay:
    return HitDisplay(
        label_of=label_of,
        hadamards=np.arange(20.0).reshape(5, 4) + 1 if hadamards is None else hadamards,
        shape=(2, 2),
        report_clusters={0},
    )


def test_merges_that_tie_share_one_step():
    assert step_ends(np.array([0.1, 0.1, 0.5, 0.5, 0.5, 0.9])) == [2, 5, 6]
    assert step_ends(np.zeros(0)) == []


def test_the_page_holds_both_hierarchies():
    data = persistence_data(two_hills(), display_of())
    assert [p.name for p in data.trees.peaks] == ["T0", "T1"]
    assert [(m.dying, m.surviving) for m in data.trees.merges] == [(1, 0)]
    assert data.trees.merges[0].persistence == 3.0
    assert data.trees.step_ends == [1]
    assert [p.hits for p in data.trees.peaks] == [3, 3]
    assert [m.persistence for m in data.modes.merges] == [0.5, 3.0]
    assert data.modes.step_ends == [1, 2]


def tokens_of_peak(data, position: int) -> list[str]:
    return [e.token for e in data.trees.peaks[position].examples]


def test_each_peak_lists_its_densest_hits_with_the_peak_first():
    data = persistence_data(two_hills(), display_of())
    assert tokens_of_peak(data, 0) == ["p0", "p1"]
    assert tokens_of_peak(data, 1) == ["p4", "p2", "p3"]
    assert data.tile_height == 2 and data.tile_width == 2


def test_identical_rows_are_one_example():
    hadamards = np.arange(20.0).reshape(5, 4) + 1
    hadamards[2] = hadamards[4]
    data = persistence_data(two_hills(), display_of(hadamards))
    assert tokens_of_peak(data, 1) == ["p4", "p3"]


def test_a_merge_between_trees_shows_its_route():
    data = persistence_data(two_hills(), display_of())
    route = data.trees.merges[0].route
    assert route is not None
    assert [s.example.token for s in route.stops] == ["p4", "p3", "p2", "p1", "p0"]
    assert [s.tree.name for s in route.stops] == ["T1", "T1", "T1", "T0", "T0"]
    assert route.merge_stop == 3
    assert route.step == 1 and route.n_points == 5
    assert all(m.route is None for m in data.modes.merges)


def test_a_tree_links_to_its_cluster_only_when_the_report_has_it():
    data = persistence_data(two_hills(), display_of())
    assert [
        (link.name, link.href) for peak in data.trees.peaks for link in peak.trees
    ] == [("T0", "index.html#cluster-0"), ("T1", None)]
    assert data.trees.merges[0].route.stops[-1].tree.href == "index.html#cluster-0"
    assert all(not peak.trees for peak in data.modes.peaks)


def test_a_thinned_route_keeps_both_roots_and_the_merge_point():
    for n_points, merge_at in [(62, 13), (21, 0), (21, 20), (5, 2), (1, 0), (100, 50)]:
        step = smallest_step(n_points, merge_at, ROUTE_STOPS)
        kept = thinned_route(n_points, merge_at, step)
        assert len(kept) <= ROUTE_STOPS
        assert kept == sorted(set(kept))
        assert {0, merge_at, n_points - 1} <= set(kept)


def test_a_short_route_is_not_thinned():
    assert smallest_step(20, 7, ROUTE_STOPS) == 1
    assert smallest_step(21, 7, ROUTE_STOPS) == 2


def test_a_tile_decodes_back_to_its_row():
    row = np.array([-2.0, 1.0, 0.0, 0.5])
    tile, largest = encoded_tile(row)
    levels = np.frombuffer(base64.b64decode(tile), dtype=np.int8)
    assert largest == 2.0
    assert np.allclose(levels / TILE_LEVELS * largest, row, atol=largest / TILE_LEVELS)


def test_an_all_zero_tile_stays_zero():
    tile, largest = encoded_tile(np.zeros(4))
    assert largest == 0.0
    assert base64.b64decode(tile) == bytes(4)


def test_only_a_top_tie_of_more_than_k_points_is_a_capped_density():
    data = two_hills()
    assert not points_at_capped_density(data).any()
    data.rho = np.array([5.0, 5.0, 3.0, 2.5, 4.0])
    assert points_at_capped_density(data).tolist() == [True, True, False, False, False]


def test_hit_text_cannot_close_the_script_tag(tmp_path):
    path = write_persistence_page(tmp_path, 7, two_hills(), display_of())
    html = path.read_text()
    match = re.search(
        r'<script id="persistence-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    assert (
        json.loads(match.group(1))["trees"]["peaks"][0]["examples"][0]["before"]
        == "</script><b>"
    )
    assert path.name == "persistence.html"
