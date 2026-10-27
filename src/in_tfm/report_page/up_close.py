"""A slice of the real forest laid out on real distances, for the "Forest up close" tab.

- Each tree contributes a few random points, plus every point on their path up to the root.
- The layout is a spring layout on each point's nearest neighbours in the slice, plus its parent link.
  - So near means similar, and positions stay fixed as trees are switched on and off.
- It runs once when the report is written, since the page only has the forest, not the rows.
"""

import networkx as nx
import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from scipy.spatial.distance import cdist

from ..density_peaks import PeakDistance
from ..weighted_jaccard import weighted_jaccard_block

POINT_BUDGET = 1500
"""Points sampled over all trees before ancestors are added. The layout is quadratic in points."""
MAX_PER_TREE = 60
MIN_PER_TREE = 5
NEIGHBOURS = 6
PARENT_WEIGHT = 2.0
"""Parent links pull harder than neighbour links, so each tree's paths stay readable."""


class UpCloseLayout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Int[np.ndarray, "n_shown"]
    """Distinct points, ascending."""
    xy: Float[np.ndarray, "n_shown 2"]
    per_tree: int


def sampled_with_ancestors(
    tree: Int[np.ndarray, "n_distinct"], parent_in_tree: Int[np.ndarray, "n_distinct"], per_tree: int, seed: int,
) -> Int[np.ndarray, "n_shown"]:
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for t in np.unique(tree):
        members = np.flatnonzero(tree == t)
        for start in rng.choice(members, min(per_tree, len(members)), replace=False):
            point = int(start)
            while point >= 0 and point not in chosen:
                chosen.add(point)
                point = int(parent_in_tree[point])
    return np.array(sorted(chosen), dtype=int)


def slice_distances(rows: Float[np.ndarray, "n_shown dim"], distance: PeakDistance) -> Float[np.ndarray, "n_shown n_shown"]:
    """`rows` are already `comparable_rows`."""
    return weighted_jaccard_block(rows, rows) if distance == "weighted_jaccard" else cdist(rows, rows)


def spring_positions(
    distances: Float[np.ndarray, "n n"], parents: Int[np.ndarray, "n"], seed: int
) -> Float[np.ndarray, "n 2"]:
    """`parents` index into the slice, -1 at roots."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(distances)))
    masked = distances + np.diag(np.full(len(distances), np.inf))
    for i, nearest in enumerate(np.argsort(masked, axis=1)[:, :NEIGHBOURS]):
        for j in nearest:
            graph.add_edge(i, int(j), weight=float(max(1.0 - distances[i, j], 1e-3)))
    for i, parent in enumerate(parents):
        if parent >= 0:
            graph.add_edge(i, int(parent), weight=PARENT_WEIGHT)
    positions = nx.spring_layout(graph, weight="weight", iterations=150, seed=seed)
    return np.array([positions[i] for i in range(len(distances))])


def up_close_layout(
    distinct_rows: Float[np.ndarray, "n_distinct dim"],
    tree: Int[np.ndarray, "n_distinct"],
    parent_in_tree: Int[np.ndarray, "n_distinct"],
    distance: PeakDistance,
    seed: int = 0,
) -> UpCloseLayout:
    n_trees = len(np.unique(tree))
    per_tree = int(np.clip(POINT_BUDGET // max(n_trees, 1), MIN_PER_TREE, MAX_PER_TREE))
    points = sampled_with_ancestors(tree, parent_in_tree, per_tree, seed)
    slot = np.full(len(tree), -1)
    slot[points] = np.arange(len(points))
    parents = np.where(parent_in_tree[points] >= 0, slot[parent_in_tree[points]], -1)
    xy = spring_positions(slice_distances(distinct_rows[points], distance), parents, seed)
    return UpCloseLayout(points=points, xy=xy, per_tree=per_tree)
