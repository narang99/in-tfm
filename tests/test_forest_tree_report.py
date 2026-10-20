import json
import re

import numpy as np

from in_tfm.density_peaks import RootCuts, density_peak_forest, forest_trees
from in_tfm.forest_report import HitArrays
from in_tfm.forest_tree_report import TREE_PAGE, forest_tree_data, write_forest_tree_page


def two_blobs_with_copies() -> np.ndarray:
    rng = np.random.default_rng(0)
    inputs = np.zeros((20, 9))
    inputs[:12, :3] = rng.uniform(1.0, 1.2, (12, 3))
    inputs[12:, 6:] = rng.uniform(1.0, 1.2, (8, 3))
    return np.vstack([inputs, inputs[:1], inputs[:1]])


def forest_and_hits():
    points = two_blobs_with_copies()
    forest = density_peak_forest(points, k=3)
    trees = forest_trees(forest, RootCuts(rho=float("-inf"), delta=0.5))
    hits = HitArrays(hadamards=points, inputs=2 * points, activations=np.arange(len(points), dtype=float))
    return forest, trees, hits


def test_roots_have_no_parent_and_hold_their_whole_tree():
    forest, trees, hits = forest_and_hits()
    data = forest_tree_data(forest, trees, hits, (3, 3))
    hits_per_tree = np.bincount(trees.tree[forest.distinct_of_hit])
    assert [data.parent[root] for root in data.roots] == [-1, -1]
    assert [data.subtree_hits[root] for root in data.roots] == hits_per_tree.tolist()


def test_every_other_point_has_a_parent_one_level_up_in_its_tree():
    forest, trees, hits = forest_and_hits()
    data = forest_tree_data(forest, trees, hits, (3, 3))
    for point, parent in enumerate(data.parent):
        if parent >= 0:
            assert data.depth[parent] == data.depth[point] - 1
            assert trees.tree[parent] == trees.tree[point]


def test_copies_are_averaged_into_one_point():
    forest, trees, hits = forest_and_hits()
    data = forest_tree_data(forest, trees, hits, (3, 3))
    copied = forest.distinct_of_hit[0]
    assert data.count[copied] == 3
    assert np.isclose(data.activation[copied], (0 + 20 + 21) / 3)
    assert np.allclose(data.input[copied], 2 * two_blobs_with_copies()[0], atol=1e-4)


def test_page_carries_every_point(tmp_path):
    forest, trees, hits = forest_and_hits()
    path = write_forest_tree_page(tmp_path, forest, trees, hits, (3, 3), "Neuron 0")
    html = path.read_text()
    payload = re.search(r'<script type="application/json" id="forest-tree-data">(.*?)</script>', html, re.S).group(1)
    assert path.name == TREE_PAGE
    assert len(json.loads(payload)["parent"]) == len(forest.counts)


def test_page_is_skipped_past_the_value_cap(tmp_path):
    forest, trees, hits = forest_and_hits()
    assert write_forest_tree_page(tmp_path, forest, trees, hits, (3, 3), "Neuron 0", max_values=10) is None
    assert not (tmp_path / TREE_PAGE).exists()
