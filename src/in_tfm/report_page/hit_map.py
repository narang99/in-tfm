"""A map of a sample of points, pulled toward the peaks of the clusters that carry their probability.

- Each cluster's root is an anchor node. A point links to its nearest sampled neighbours, and to the anchor of each
  cluster it has some probability in, with that probability as the pull.
  - So a confident point sits near its peak, and a point split between two clusters sits between their peaks.
- The layout runs once when the report is written, like the up-close slice, since the page has no rows.
"""

import networkx as nx
import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from ..density_peaks import PeakDistance
from .up_close import slice_distances

MAP_POINTS = 1500
NEIGHBOURS = 6
MIN_PULL = 0.05
"""Probabilities under this get no pull, so a point is not tied to every cluster."""
PULL_WEIGHT = 3.0
ITERATIONS = 100


class HitMapLayout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Int[np.ndarray, "n_shown"]
    """Distinct points, ascending. Each tree's root is always among them."""
    xy: Float[np.ndarray, "n_shown 2"]


def sampled_points(
    counts: Int[np.ndarray, "n_distinct"], roots: Int[np.ndarray, "n_trees"], seed: int
) -> Int[np.ndarray, "n_shown"]:
    """Weighted by copies, so the map shows hits in proportion, plus every root as its cluster's anchor."""
    n = len(counts)
    size = min(MAP_POINTS, n)
    picked = np.random.default_rng(seed).choice(n, size=size, replace=False, p=counts / counts.sum())
    return np.unique(np.concatenate([picked, roots]))


def map_positions(
    distances: Float[np.ndarray, "n n"],
    probabilities: Float[np.ndarray, "n n_trees"],
    anchor_of_tree: Int[np.ndarray, "n_trees"],
    seed: int,
) -> Float[np.ndarray, "n 2"]:
    """`anchor_of_tree` indexes into the sample: the slot of each tree's root."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(distances)))
    masked = distances + np.diag(np.full(len(distances), np.inf))
    for i, nearest in enumerate(np.argsort(masked, axis=1)[:, :NEIGHBOURS]):
        for j in nearest:
            graph.add_edge(i, int(j), weight=float(max(1.0 - distances[i, j], 1e-3)))
    for i, row in enumerate(probabilities):
        for tree in np.flatnonzero(row >= MIN_PULL):
            anchor = int(anchor_of_tree[tree])
            if anchor != i:
                graph.add_edge(i, anchor, weight=PULL_WEIGHT * float(row[tree]))
    positions = nx.spring_layout(graph, weight="weight", iterations=ITERATIONS, seed=seed)
    return np.array([positions[i] for i in range(len(distances))])


def hit_map_layout(
    distinct_rows: Float[np.ndarray, "n_distinct dim"],
    counts: Int[np.ndarray, "n_distinct"],
    roots: Int[np.ndarray, "n_trees"],
    probabilities: Float[np.ndarray, "n_distinct n_trees"],
    distance: PeakDistance,
    seed: int = 0,
) -> HitMapLayout:
    points = sampled_points(counts, roots, seed)
    slot = np.full(len(counts), -1)
    slot[points] = np.arange(len(points))
    xy = map_positions(
        slice_distances(distinct_rows[points], distance), probabilities[points], slot[roots], seed,
    )
    return HitMapLayout(points=points, xy=xy)
