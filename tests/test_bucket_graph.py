import networkx as nx
import numpy as np

from in_tfm.bucket_graph import (
    connected_neighbour_graph, layout, link_to_peaks, neighbour_graph, peak_hits, sample_hits, sample_map,
)
from in_tfm.hit_buckets import Bucket
from in_tfm.run_config import BucketMapConfig

CONFIG = BucketMapConfig()


def test_a_peak_is_the_first_hit_of_the_root_point():
    assert peak_hits(np.array([0, 1, 1, 2]), np.array([2, 1])) == [3, 1]


def test_the_sample_has_the_peaks_and_spread_hits_of_each_paged_bucket():
    probabilities = np.tile(np.linspace(0.1, 0.9, 20)[:, None], (1, 2))
    paged = Bucket(name="big", clusters=(0,), hits=list(range(20)))
    sample = sample_hits([paged], probabilities, peaks=[5], per_bucket=3)
    assert sample.tolist() == [0, 5, 10, 19]


def test_each_node_links_to_its_nearest_in_either_direction():
    distances = np.array([[0, 0.1, 0.9], [0.1, 0, 0.8], [0.9, 0.8, 0]])
    graph = neighbour_graph(distances, 1)
    assert sorted(graph.edges) == [(0, 1), (1, 2)]
    assert np.isclose(graph[0][1]["weight"], 0.9)


def test_more_neighbours_are_tried_until_the_sample_is_connected():
    two_islands = np.full((40, 40), 0.9)
    two_islands[:20, :20] = two_islands[20:, 20:] = 0.1
    graph, k = connected_neighbour_graph(two_islands, CONFIG.neighbour_counts)
    assert nx.is_connected(graph)
    assert k > 6


def test_the_layout_keeps_linked_nodes_closer_than_unlinked_ones():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    points = layout(graph)
    assert np.linalg.norm(points[0] - points[1]) < np.linalg.norm(points[0] - points[3])


def test_a_hit_is_pulled_toward_each_peak_it_has_a_real_probability_for():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.4]])
    link_to_peaks(graph, probabilities, peak_nodes=[0, 1], config=CONFIG)
    assert sorted(graph.edges) == [(0, 2), (1, 2)]
    assert graph[0][2]["weight"] > graph[1][2]["weight"]


def test_a_pull_adds_to_the_neighbour_link_between_the_same_nodes():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.5)
    link_to_peaks(graph, np.array([[1.0], [1.0]]), peak_nodes=[0], config=CONFIG)
    assert graph[0][1]["weight"] > 0.5


def test_the_sample_map_puts_each_cluster_peak_on_the_map_and_pulls_hits_to_it():
    rng = np.random.default_rng(0)
    n = 30
    labels = np.repeat([0, 1], n // 2)
    probabilities = np.eye(2)[labels] * 0.9 + 0.05
    distances = np.where(labels[:, None] == labels[None], 0.2, 0.8) + rng.uniform(0, 0.05, (n, n))
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    result = sample_map(distances, probabilities, np.arange(n), peaks=[0, n - 1], config=CONFIG)
    assert result.peak_nodes == {0: 0, 1: n - 1}
    assert result.labels.tolist() == labels.tolist()
    assert result.points.shape == (n, 2)
    one, other = result.points[labels == 0].mean(axis=0), result.points[labels == 1].mean(axis=0)
    assert np.linalg.norm(one - other) > 0.5
