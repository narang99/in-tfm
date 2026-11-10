"""ToMATo's merge hierarchy (Chazal et al., 2013), to tell separate peaks from bumps on one hill.

- Points enter from densest to sparsest, over the same either-way kNN graph that memberships use.
- A new point with no denser neighbour starts a component, with itself as the peak.
- A point joining two components is where they meet, its saddle.
  - The lower peak dies there, and its persistence is its density minus the saddle's.
- Merging every peak of persistence up to a threshold gives ToMATo's clusters at that threshold.
- `merge_routes` shows how a merge happened, as a graph path between the two peaks through the saddle.
  - It is not a forest branch, since a saddle often belongs to a third tree that merged earlier.
- Points of equal density joined by an edge count as one node.
  - IVF keeps copies, and copies tie. Breaking those ties by index would turn most peaks into slopes.
"""

import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from scipy.sparse.csgraph import breadth_first_order, connected_components


class PeakMerges(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    peaks: Int[np.ndarray, "n_peaks"]
    """One point per peak, the lowest index on its plateau."""
    peak_of_point: Int[np.ndarray, "n_points"]
    """Position in `peaks` of the point's basin. -1 when its component never met a peak."""
    dying: Int[np.ndarray, "n_merges"]
    """Position in `peaks`. Merges run from the lowest persistence up."""
    surviving: Int[np.ndarray, "n_merges"]
    """The higher peak of the two components when they met. It may die later itself."""
    saddle: Int[np.ndarray, "n_merges"]
    """The point where the two components met."""
    persistence: Float[np.ndarray, "n_merges"]


class MergeRoute(BaseModel):
    points: list[int]
    """From the dying peak, through the saddle, to the surviving peak."""
    merge_at: int
    """Position of the saddle in `points`."""


class PlateauGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_of_point: Int[np.ndarray, "n_points"]
    first_point: Int[np.ndarray, "n_nodes"]
    density: Float[np.ndarray, "n_nodes"]
    adjacency: sp.csr_matrix


class SweepResult(BaseModel):
    """Per node, or per merge."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    basin: list[int]
    dying: list[int]
    surviving: list[int]
    saddle: list[int]


def plateau_graph(
    rho: Float[np.ndarray, "n_points"], neighbours: Int[np.ndarray, "n_points k"]
) -> PlateauGraph:
    n = len(rho)
    sources = np.repeat(np.arange(n), neighbours.shape[1])
    targets = neighbours.ravel()
    ties = rho[sources] == rho[targets]
    tied = sp.coo_matrix(
        (np.ones(int(ties.sum()), dtype=np.int8), (sources[ties], targets[ties])),
        shape=(n, n),
    )
    _, node_of_point = connected_components(tied, directed=False)
    _, first_point = np.unique(node_of_point, return_index=True)
    a, b = node_of_point[sources], node_of_point[targets]
    keep = a != b
    n_nodes = len(first_point)
    both_ways = sp.coo_matrix(
        (
            np.ones(2 * int(keep.sum()), dtype=np.int8),
            (
                np.concatenate([a[keep], b[keep]]),
                np.concatenate([b[keep], a[keep]]),
            ),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    both_ways.sum_duplicates()
    return PlateauGraph(
        node_of_point=node_of_point,
        first_point=first_point,
        density=rho[first_point],
        adjacency=both_ways,
    )


def density_order(density: Float[np.ndarray, "n_nodes"]) -> Int[np.ndarray, "n_nodes"]:
    """Densest first. Lower index first among nodes that tie without an edge between them."""
    return np.lexsort((np.arange(len(density)), -density))


def entry_rank(order: Int[np.ndarray, "n_nodes"]) -> Int[np.ndarray, "n_nodes"]:
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def denser_neighbours(
    graph: PlateauGraph, rank: Int[np.ndarray, "n_nodes"]
) -> list[list[int]]:
    """Each node's neighbours that entered before it, the densest first."""
    pointers, indices = graph.adjacency.indptr, graph.adjacency.indices
    result = []
    for node in range(len(rank)):
        listed = indices[pointers[node] : pointers[node + 1]]
        denser = listed[rank[listed] < rank[node]]
        result.append(denser[np.argsort(rank[denser])].tolist())
    return result


def find_owner(owner: list[int], node: int) -> int:
    while owner[node] != node:
        owner[node] = owner[owner[node]]
        node = owner[node]
    return node


def sweep(
    order: Int[np.ndarray, "n_nodes"],
    rank: Int[np.ndarray, "n_nodes"],
    denser: list[list[int]],
    is_peak: Bool[np.ndarray, "n_nodes"],
) -> SweepResult:
    """A component's head is its highest peak, or its densest node until a peak joins it.

    - Two peaks meeting is a merge.
    - Any other meeting is silent, and the node without peak status hands its basin to the other head."""
    n = len(order)
    owner, head, adopted = list(range(n)), list(range(n)), list(range(n))
    basin = list(range(n))
    dying: list[int] = []
    surviving: list[int] = []
    saddle: list[int] = []
    peak_flags, ranks = is_peak.tolist(), rank.tolist()
    for node in order.tolist():
        if denser[node] and not peak_flags[node]:
            basin[node] = head[find_owner(owner, denser[node][0])]
        for other in denser[node]:
            a, b = find_owner(owner, node), find_owner(owner, other)
            if a == b:
                continue
            head_a, head_b = head[a], head[b]
            a_wins = (not peak_flags[head_a], ranks[head_a]) < (
                not peak_flags[head_b],
                ranks[head_b],
            )
            top, low = (a, b) if a_wins else (b, a)
            top_head, low_head = head[top], head[low]
            if peak_flags[top_head] and peak_flags[low_head]:
                dying.append(low_head)
                surviving.append(top_head)
                saddle.append(node)
            else:
                adopted[low_head] = top_head
            owner[low] = top
    for node in range(n):
        here = basin[node]
        while adopted[here] != here:
            here = adopted[here]
        basin[node] = here
    return SweepResult(basin=basin, dying=dying, surviving=surviving, saddle=saddle)


def peak_nodes(
    graph: PlateauGraph,
    denser: list[list[int]],
    seeds: Int[np.ndarray, "n_seeds"] | None,
) -> Bool[np.ndarray, "n_nodes"]:
    if seeds is None:
        return np.array([not listed for listed in denser], dtype=bool)
    is_peak = np.zeros(len(denser), dtype=bool)
    is_peak[graph.node_of_point[seeds]] = True
    return is_peak


def peak_merges(
    rho: Float[np.ndarray, "n_points"],
    neighbours: Int[np.ndarray, "n_points k"],
    seeds: Int[np.ndarray, "n_seeds"] | None = None,
) -> PeakMerges:
    """Every local mode is a peak by default, as in ToMATo.

    With `seeds` only those points are peaks, so the merges are between them alone.
    - A seed on another peak's slope merges where it enters, with persistence 0."""
    graph = plateau_graph(rho, neighbours)
    order = density_order(graph.density)
    rank = entry_rank(order)
    denser = denser_neighbours(graph, rank)
    is_peak = peak_nodes(graph, denser, seeds)
    result = sweep(order, rank, denser, is_peak)

    nodes = np.flatnonzero(is_peak)
    position = np.full(len(order), -1, dtype=np.int64)
    position[nodes] = np.arange(len(nodes))
    dying_nodes = np.array(result.dying, dtype=np.int64)
    saddle_nodes = np.array(result.saddle, dtype=np.int64)
    persistence = graph.density[dying_nodes] - graph.density[saddle_nodes]
    by_persistence = np.argsort(persistence, kind="stable")
    return PeakMerges(
        peaks=graph.first_point[nodes],
        peak_of_point=position[np.array(result.basin)][graph.node_of_point],
        dying=position[dying_nodes][by_persistence],
        surviving=position[np.array(result.surviving, dtype=np.int64)][by_persistence],
        saddle=graph.first_point[saddle_nodes][by_persistence],
        persistence=persistence[by_persistence],
    )


def paths_from_saddle(
    graph: PlateauGraph,
    rank: Int[np.ndarray, "n_nodes"],
    saddle: int,
    targets: list[int],
) -> list[list[int]]:
    """Fewest hops over the nodes that had entered by the merge, the only ones that joined anything yet.

    - The two sides of a merge were separate components until the saddle entered.
      - So their paths share only the saddle.
    """
    entered = np.flatnonzero(rank <= rank[saddle])
    position = np.full(len(rank), -1, dtype=np.int64)
    position[entered] = np.arange(len(entered))
    _, previous = breadth_first_order(
        graph.adjacency[entered][:, entered],
        position[saddle],
        return_predecessors=True,
    )
    paths = []
    for target in targets:
        here = int(position[target])
        path = [target]
        while here != position[saddle]:
            here = int(previous[here])
            path.append(int(entered[here]))
        paths.append(path[::-1])
    return paths


def merge_routes(
    rho: Float[np.ndarray, "n_points"],
    neighbours: Int[np.ndarray, "n_points k"],
    merges: PeakMerges,
) -> list[MergeRoute]:
    """One route per merge, in the order of `merges`.

    - It can pass through other peaks' hits and dip below them, when those peaks already joined a side.
    - Each search covers every node that had entered, so this is for tens of merges, not thousands.
    """
    graph = plateau_graph(rho, neighbours)
    rank = entry_rank(density_order(graph.density))
    routes = []
    for dying, surviving, saddle in zip(
        merges.peaks[merges.dying].tolist(),
        merges.peaks[merges.surviving].tolist(),
        merges.saddle.tolist(),
    ):
        to_dying, to_surviving = paths_from_saddle(
            graph,
            rank,
            int(graph.node_of_point[saddle]),
            [int(graph.node_of_point[dying]), int(graph.node_of_point[surviving])],
        )
        nodes = to_dying[::-1] + to_surviving[1:]
        routes.append(
            MergeRoute(
                points=graph.first_point[nodes].tolist(),
                merge_at=len(to_dying) - 1,
            )
        )
    return routes


def clusters_at(merges: PeakMerges, threshold: float) -> Int[np.ndarray, "n_points"]:
    """Each point's surviving peak, as a position in `peaks`, after merging up to `threshold`."""
    final = np.arange(len(merges.peaks))
    owner = final.tolist()
    # Merges run from low persistence up, so a dying peak's survivor may be merged by a later row.
    for dying, surviving in zip(
        merges.dying[merges.persistence <= threshold].tolist(),
        merges.surviving[merges.persistence <= threshold].tolist(),
    ):
        owner[dying] = surviving
    final = np.array(
        [find_owner(owner, peak) for peak in final.tolist()], dtype=np.int64
    )
    clusters = np.full(len(merges.peak_of_point), -1, dtype=np.int64)
    assigned = merges.peak_of_point >= 0
    clusters[assigned] = final[merges.peak_of_point[assigned]]
    return clusters
