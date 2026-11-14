import hdbscan
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from in_tfm.hdbscan_merges import fit_tree, merge_nodes, root_of, shown_hits


@pytest.fixture
def distances() -> np.ndarray:
    """Two pairs of blobs: each pair joins first, then the pairs join."""
    rng = np.random.default_rng(0)
    centres = [(0, 0), (3, 0), (12, 0), (15, 0)]
    points = np.vstack([rng.normal(c, 0.5, (60, 2)) for c in centres])
    return cdist(points, points)


def test_every_hit_is_under_the_root_once(distances: np.ndarray) -> None:
    nodes = merge_nodes(fit_tree(distances, 10, None))
    root = nodes[root_of(nodes)]
    assert np.array_equal(root.members, np.arange(len(distances)))
    dropped = np.concatenate([n.dropped for n in nodes.values()])
    assert len(dropped) == len(set(dropped.tolist())) == len(distances)


def test_leaves_are_hdbscan_s_leaf_clusters(distances: np.ndarray) -> None:
    nodes = merge_nodes(fit_tree(distances, 10, None))
    leaf_fit = hdbscan.HDBSCAN(
        metric="precomputed", min_cluster_size=10, cluster_selection_method="leaf"
    ).fit(distances)
    expected = {int(n) for n in leaf_fit.condensed_tree_._select_clusters()}
    assert {n for n, node in nodes.items() if node.is_leaf} == expected


def test_merges_join_closer_than_their_parent(distances: np.ndarray) -> None:
    nodes = merge_nodes(fit_tree(distances, 10, None))
    for node in nodes.values():
        if node.merge_distance is not None and node.parent is not None:
            assert node.merge_distance < node.birth_distance
        for child in node.children:
            assert set(nodes[child].members) <= set(node.members)


def test_a_merge_card_shows_both_sides(distances: np.ndarray) -> None:
    nodes = merge_nodes(fit_tree(distances, 10, None))
    root = root_of(nodes)
    shown = shown_hits(nodes, root, per_card=4)
    for child in nodes[root].children:
        assert set(shown) & set(nodes[child].members.tolist())


def test_a_card_shows_one_hit_per_copy_group(distances: np.ndarray) -> None:
    copies = np.zeros((20, 20))
    padded = np.pad(distances, ((0, 20), (0, 20)), constant_values=40.0)
    padded[-20:, -20:] = copies
    nodes = merge_nodes(fit_tree(padded, 10, None))
    copy_of = np.concatenate([np.arange(len(distances)), np.full(20, len(distances))])
    for node in nodes:
        shown = shown_hits(nodes, node, per_card=6, copy_of=copy_of)
        assert len(set(copy_of[shown])) == len(shown)


def test_the_graph_puts_each_merge_between_and_above_its_groups(
    distances: np.ndarray,
) -> None:
    from in_tfm.merge_page import node_heights, x_positions

    nodes = merge_nodes(fit_tree(distances, 10, None))
    x, height = x_positions(nodes), node_heights(nodes)
    leaves = [x[n] for n, node in nodes.items() if node.is_leaf]
    assert len(set(leaves)) == len(leaves)
    for n, node in nodes.items():
        if node.children:
            xs = [x[c] for c in node.children]
            assert min(xs) <= x[n] <= max(xs)
            assert all(height[n] > height[c] for c in node.children)


def test_trimming_keeps_every_leaf_under_one_eom_cluster(
    distances: np.ndarray,
) -> None:
    from in_tfm.hdbscan_merges import leaf_and_eom_only

    nodes = merge_nodes(fit_tree(distances, 10, None))
    trimmed = leaf_and_eom_only(nodes)
    assert all(n.is_leaf or n.is_eom for n in trimmed.values())
    assert {n for n, node in nodes.items() if node.is_leaf} <= set(trimmed)
    for node in trimmed.values():
        assert (node.parent is None) == node.is_eom
        if node.parent is not None:
            assert trimmed[node.parent].is_eom
            assert set(node.members) <= set(trimmed[node.parent].members)
