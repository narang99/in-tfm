"""A neuron's hits with their density-peak forest, saved together, and walks over that forest's graph.

- The report keeps only a few hits per cluster, so a new vector cannot be placed among all hits from it.
  - `HitSpace` keeps every hit's Hadamard vector next to the forest, so a vector from anywhere can be placed.
- Distances are the forest's own: `comparable_rows` with the run's distance and normalising.
- Walks run over the forest's distinct points, since the graph and the trees are built on those.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from scipy.sparse.csgraph import dijkstra

from .density_peaks.forest import DensityPeakForest, ForestTrees, PeakDistance, RootCuts, comparable_rows, merge_copies
from .density_peaks.probabilities import knn_graph, peak_cores, step_length
from .weighted_jaccard.vectorized import weighted_jaccard_block

FOREST_FIELDS = (
    "distinct_of_hit", "counts", "kth_distance", "density_rank", "parent", "delta", "neighbours", "neighbour_distance",
)


class HitSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hadamard: Float[np.ndarray, "n_hits dim"]
    activation: Float[np.ndarray, "n_hits"]
    sample_ids: list[str]
    """The scan's samples. `batch_idx` indexes into it."""
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    grid_width: int
    threshold: float
    distance: PeakDistance
    normalise: bool
    peak_share: float
    forest: DensityPeakForest
    trees: ForestTrees

    def save(self, path: Path) -> None:
        np.savez(
            path,
            hadamard=self.hadamard, activation=self.activation, sample_ids=np.array(self.sample_ids),
            batch_idx=self.batch_idx, token_idx=self.token_idx, grid_width=self.grid_width, threshold=self.threshold,
            distance=self.distance, normalise=self.normalise, peak_share=self.peak_share,
            roots=self.trees.roots, tree=self.trees.tree, depth=self.trees.depth,
            cut_rho=self.trees.cuts.rho, cut_delta=self.trees.cuts.delta,
            **{f"forest_{name}": getattr(self.forest, name) for name in FOREST_FIELDS},
        )

    @classmethod
    def load(cls, path: Path) -> "HitSpace":
        data = np.load(path)
        return cls(
            hadamard=data["hadamard"], activation=data["activation"], sample_ids=data["sample_ids"].tolist(),
            batch_idx=data["batch_idx"], token_idx=data["token_idx"], grid_width=int(data["grid_width"]),
            threshold=float(data["threshold"]), distance=str(data["distance"]), normalise=bool(data["normalise"]),
            peak_share=float(data["peak_share"]),
            forest=DensityPeakForest(**{name: data[f"forest_{name}"] for name in FOREST_FIELDS}),
            trees=ForestTrees(
                cuts=RootCuts(rho=float(data["cut_rho"]), delta=float(data["cut_delta"])),
                roots=data["roots"], tree=data["tree"], depth=data["depth"],
            ),
        )


class HitGraph:
    """The forest's distinct points, ready for nearest-neighbour lookups and shortest paths."""

    def __init__(self, space: HitSpace) -> None:
        self.space = space
        self.forest = space.forest
        self.trees = space.trees
        rows = comparable_rows(space.hadamard.astype(np.float64), space.distance, space.normalise)
        self.distinct_rows, distinct_of_hit, _ = merge_copies(rows)
        if not np.array_equal(distinct_of_hit, self.forest.distinct_of_hit):
            raise ValueError("the hits do not merge into the forest's distinct points")
        self.graph: sp.csr_matrix = knn_graph(self.forest)
        self.hop_graph = self.graph.copy()
        self.hop_graph.data[:] = 1.0
        self.cores = peak_cores(self.forest, self.trees, space.peak_share)
        self.tau = step_length(self.forest)
        self.first_hit_of = np.full(len(self.distinct_rows), -1)
        for hit in range(len(distinct_of_hit) - 1, -1, -1):
            self.first_hit_of[distinct_of_hit[hit]] = hit

    def distances_to_points(self, vectors: Float[np.ndarray, "n dim"]) -> Float[np.ndarray, "n n_distinct"]:
        if self.space.distance != "weighted_jaccard":
            raise NotImplementedError("only weighted Jaccard is wired up here")
        return weighted_jaccard_block(comparable_rows(vectors.astype(np.float64), "weighted_jaccard", self.space.normalise), self.distinct_rows)

    def nearest(self, vectors: Float[np.ndarray, "n dim"]) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
        distances = self.distances_to_points(vectors)
        nearest = distances.argmin(axis=1)
        return nearest, distances[np.arange(len(nearest)), nearest]

    def nearest_by_direction(self, vectors: Float[np.ndarray, "n dim"]) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
        """The nearest point when every row is L1-normalised first, so size no longer counts.

        - The forest's own distance counts size, so a vector far larger than every hit lands on the largest hits.
          - Comparing directions tells whether its make-up still resembles a tree's, whatever its size."""
        if not hasattr(self, "direction_rows"):
            self.direction_rows = comparable_rows(self.space.hadamard[self.first_hit_of].astype(np.float64), "weighted_jaccard", True)
        distances = weighted_jaccard_block(comparable_rows(vectors.astype(np.float64), "weighted_jaccard", True), self.direction_rows)
        nearest = distances.argmin(axis=1)
        return nearest, distances[np.arange(len(nearest)), nearest]

    def ascent(self, point: int) -> list[int]:
        """Parent links from `point` up to its tree's root, the forest's own path uphill in density."""
        root = self.trees.roots[self.trees.tree[point]]
        path = [point]
        while path[-1] != root:
            path.append(int(self.forest.parent[path[-1]]))
        return path

    def core_points(self, trees: list[int]) -> Int[np.ndarray, "n_core"]:
        return np.concatenate([self.cores[tree] for tree in trees])

    def shortest_path_to(self, start: int, targets: Int[np.ndarray, "n_targets"]) -> tuple[list[int], float]:
        """The geodesic over the kNN graph from `start` to the closest of `targets`, and its length."""
        distances, predecessors = dijkstra(self.graph, directed=False, indices=start, return_predecessors=True)
        target = int(targets[np.argmin(distances[targets])])
        if not np.isfinite(distances[target]):
            return [start], float("inf")
        path = [target]
        while path[-1] != start:
            path.append(int(predecessors[path[-1]]))
        return path[::-1], float(distances[target])

    def geodesic_to(self, points: Int[np.ndarray, "n"], targets: Int[np.ndarray, "n_targets"]) -> Float[np.ndarray, "n"]:
        unique, inverse = np.unique(points, return_inverse=True)
        from_targets = dijkstra(self.graph, directed=False, indices=targets, min_only=True)
        return from_targets[unique][inverse]

    def hops_between(self, sources: Int[np.ndarray, "n"], targets: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
        """Edges on the fewest-edge path between each pair. -1 when no path joins them."""
        hops = np.empty(len(sources), dtype=int)
        for i, (source, target) in enumerate(zip(sources, targets)):
            if source == target:
                hops[i] = 0
                continue
            length = dijkstra(self.hop_graph, directed=False, indices=int(source), unweighted=True)[target]
            hops[i] = int(length) if np.isfinite(length) else -1
        return hops

    def hit_position(self, point: int) -> tuple[str, int, int]:
        """The sample, row and column of the first hit that became `point`."""
        hit = self.first_hit_of[point]
        row, col = divmod(int(self.space.token_idx[hit]), self.space.grid_width)
        return self.space.sample_ids[self.space.batch_idx[hit]], row, col
