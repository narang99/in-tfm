"""A sample of the hits, joined by their nearest neighbours and laid out as a graph.

- The sample takes evenly spread hits from every bucket, plus each cluster's peak so every core is on the map.
- Edges are nearest neighbours in the real distance, so a hit's local neighbourhood stays together.
- Each hit is also pulled toward the peak of every cluster it has a real probability for.
  - Without it the layout is one mixed blob, since neighbours of a mixed hit belong to many clusters.
  - A hit split between two clusters sits between their peaks, which is where the clusters meet.
- The layout only uses those edges, so no feature is reduced to two dimensions.
"""

import networkx as nx
import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from .hit_buckets import Bucket
from .pair_mixed_report import spread_by_share
from .run_config import BucketMapConfig

LAYOUT_ITERATIONS = 300
LAYOUT_SEED = 0


def peak_hits(distinct_of_hit: Int[np.ndarray, "n_hits"], roots: Int[np.ndarray, "n_clusters"]) -> list[int]:
    """The first hit of each cluster's root point, in cluster order."""
    return [int(np.flatnonzero(distinct_of_hit == root)[0]) for root in roots]


def sample_hits(
    paged: list[Bucket], probabilities: Float[np.ndarray, "n_hits n_clusters"], peaks: list[int], per_bucket: int,
) -> Int[np.ndarray, "n_sampled"]:
    """Hits of each bucket at evenly spaced ranks of its leading cluster's probability, so a pair runs from one side to the other."""
    sampled = set(peaks)
    for bucket in paged:
        hits = np.array(bucket.hits)
        picked = spread_by_share(hits, probabilities[hits, bucket.clusters[0]], per_bucket)
        sampled.update(picked.tolist())
    return np.array(sorted(sampled))


def neighbour_graph(distances: Float[np.ndarray, "m m"], k: int) -> nx.Graph:
    """Each node links to its k nearest, so a link in either direction is an edge. Weight is similarity, `1 - distance`."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(distances)))
    masked = np.array(distances, dtype=np.float64)
    np.fill_diagonal(masked, np.inf)
    nearest = np.argsort(masked, axis=1)[:, :k]
    for node, others in enumerate(nearest):
        for other in others:
            graph.add_edge(node, int(other), weight=float(1.0 - masked[node, other]))
    return graph


def connected_neighbour_graph(distances: Float[np.ndarray, "m m"], neighbour_counts: list[int]) -> tuple[nx.Graph, int]:
    """The graph with the fewest neighbours that is one component, or the densest tried if none is.

    - A separate island would otherwise be placed at random, which reads as distance that is not there."""
    for k in neighbour_counts:
        graph = neighbour_graph(distances, min(k, len(distances) - 1))
        if nx.is_connected(graph):
            return graph, k
    return graph, k


def link_to_peaks(
    graph: nx.Graph, probabilities: Float[np.ndarray, "m n_clusters"], peak_nodes: list[int], config: BucketMapConfig,
) -> None:
    """`peak_nodes[c]` is the node of cluster c's peak. Links add to an existing edge between the same two nodes."""
    for node, cluster in zip(*np.nonzero(probabilities >= config.min_pull_probability)):
        peak = peak_nodes[cluster]
        if peak == node:
            continue
        pull = config.peak_pull * probabilities[node, cluster]
        graph.add_edge(int(node), peak, weight=pull + (graph[node][peak]["weight"] if graph.has_edge(node, peak) else 0.0))


def layout(graph: nx.Graph) -> Float[np.ndarray, "m 2"]:
    positions = nx.spring_layout(graph, weight="weight", iterations=LAYOUT_ITERATIONS, seed=LAYOUT_SEED)
    return np.array([positions[node] for node in range(graph.number_of_nodes())])


class SampleMap(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hits: Int[np.ndarray, "m"]
    """The sampled hit of each node, ascending."""
    points: Float[np.ndarray, "m 2"]
    labels: Int[np.ndarray, "m"]
    """Each node's most probable cluster."""
    peak_nodes: dict[int, int]
    """Cluster id to the node of its peak."""
    neighbours: int


def sample_map(
    distances: Float[np.ndarray, "n n"], probabilities: Float[np.ndarray, "n n_clusters"], sample: Int[np.ndarray, "m"],
    peaks: list[int], config: BucketMapConfig,
) -> SampleMap:
    graph, neighbours = connected_neighbour_graph(distances[np.ix_(sample, sample)], config.neighbour_counts)
    node_of_hit = {int(hit): node for node, hit in enumerate(sample)}
    peak_nodes = {cluster: node_of_hit[peak] for cluster, peak in enumerate(peaks)}
    sampled_probabilities = probabilities[sample]
    link_to_peaks(graph, sampled_probabilities, [peak_nodes[c] for c in range(len(peaks))], config)
    return SampleMap(
        hits=sample, points=layout(graph), labels=sampled_probabilities.argmax(axis=1), peak_nodes=peak_nodes,
        neighbours=neighbours,
    )
