"""`geodesic_entropy` for a batch of trees at a time, as frontier relaxation in torch, so it runs on a GPU.

- The scipy version runs one Dijkstra per tree on CPU cores.
  - On head 0 at 32M points, each point was within the margin of a large share of about 6,000 trees.
  - That took 56 minutes on 12 cores. The work is real, but it is the same simple step many times over.
- Here a batch of trees keeps a dense distance array, trees x points, in float32.
  - The tree's peak points start at 0. Each round, the points whose distance just fell relax their edges.
  - A candidate past the margin is dropped. Lengths never shrink along a path, so nothing it leads to is kept.
  - Rounds end when no distance falls. That is a label-correcting search, and it ends at the shortest paths.
- Edges are `_past_nearest_peak` edges in steps of τ, as in the scipy version, so the two agree up to float32.
- `tests/test_density_peak_probabilities.py` checks it against `geodesic_entropy` on torch's CPU device.
"""

import numpy as np
import scipy.sparse as sp
import torch
from jaxtyping import Float, Int
from scipy.sparse.csgraph import dijkstra

from ..forest import DensityPeakForest, ForestTrees
from .vectorized import _past_nearest_peak, knn_graph, peak_cores, step_length

STATE_ELEMENTS = 2**31
"""Distances held at once, trees x points. 8 GB of float32."""
EDGES_PER_STEP = 2**27
"""Edges relaxed in one step of a round."""


class CsrGraph:
    def __init__(self, graph: sp.csr_matrix, device: str):
        self.n = graph.shape[0]
        self.indptr = torch.from_numpy(graph.indptr.astype(np.int64)).to(device)
        self.indices = torch.from_numpy(graph.indices.astype(np.int64)).to(device)
        self.lengths = torch.from_numpy(graph.data.astype(np.float32)).to(device)
        self.degree = self.indptr[1:] - self.indptr[:-1]


def frontier_geodesic_entropy(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float, margin_steps: float, device: str = "cuda",
) -> Float[np.ndarray, "n_distinct"]:
    graph = knn_graph(forest)
    cores = peak_cores(forest, trees, peak_share)
    tau = step_length(forest)
    nearest = dijkstra(graph, directed=False, indices=np.concatenate(cores), min_only=True) / tau
    steps = CsrGraph(_past_nearest_peak(graph / tau, nearest), device)
    total_weight = torch.zeros(steps.n, dtype=torch.float64, device=device)
    weighted_steps = torch.zeros(steps.n, dtype=torch.float64, device=device)
    batch = max(1, STATE_ELEMENTS // max(1, steps.n))
    for first in range(0, len(cores), batch):
        distance = batch_distances(steps, cores[first:first + batch], margin_steps, device)
        add_weights(distance, total_weight, weighted_steps)
        del distance
    nats = torch.zeros(steps.n, dtype=torch.float64, device=device)
    reachable = total_weight > 0
    nats[reachable] = torch.log(total_weight[reachable]) + weighted_steps[reachable] / total_weight[reachable]
    return (nats / np.log(2)).cpu().numpy()


def add_weights(
    distance: Float[torch.Tensor, "batch n"], total_weight: Float[torch.Tensor, "n"],
    weighted_steps: Float[torch.Tensor, "n"], rows: int = 16,
) -> None:
    """Dense, a few trees at a time, since indexing the reached states took more memory than the states.

    - exp(-inf) is 0, so a tree out of reach adds nothing. Its steps are zeroed so inf * 0 makes no NaN."""
    for start in range(0, len(distance), rows):
        past = distance[start:start + rows]
        weight = torch.exp(-past)
        total_weight += weight.sum(dim=0, dtype=torch.float64)
        weighted_steps += (weight * torch.nan_to_num(past, posinf=0.0)).sum(dim=0, dtype=torch.float64)


def batch_distances(
    steps: CsrGraph, cores: list[Int[np.ndarray, "n_peak"]], margin_steps: float, device: str
) -> Float[torch.Tensor, "batch n"]:
    """Each tree's steps past the nearest peak, inf past the margin. Flat index t * n + point is one state."""
    distance = torch.full((len(cores), steps.n), torch.inf, dtype=torch.float32, device=device)
    flat = distance.view(-1)
    sources = torch.from_numpy(np.concatenate([t * steps.n + core for t, core in enumerate(cores)])).to(device)
    flat[sources] = 0.0
    frontier = sources
    while len(frontier):
        changed = [relax(steps, flat, part, margin_steps) for part in frontier_parts(steps, frontier)]
        frontier = torch.unique(torch.cat(changed))
    return distance


def frontier_parts(steps: CsrGraph, frontier: Int[torch.Tensor, "f"]) -> list[Int[torch.Tensor, "p"]]:
    """Splits the frontier so each part relaxes about EDGES_PER_STEP edges."""
    edges = torch.cumsum(steps.degree[frontier % steps.n], 0)
    if len(edges) == 0 or edges[-1] <= EDGES_PER_STEP:
        return [frontier]
    cuts = torch.searchsorted(edges, torch.arange(EDGES_PER_STEP, int(edges[-1]), EDGES_PER_STEP, device=edges.device))
    return list(torch.tensor_split(frontier, cuts.cpu()))


def relax(
    steps: CsrGraph, flat: Float[torch.Tensor, "states"], states: Int[torch.Tensor, "p"], margin_steps: float
) -> Int[torch.Tensor, "c"]:
    """Relaxes every edge out of `states`, and returns the states whose distance fell."""
    tree, point = states // steps.n, states % steps.n
    degree = steps.degree[point]
    owner = torch.repeat_interleave(torch.arange(len(states), device=states.device), degree)
    offset = torch.arange(len(owner), device=states.device) - (torch.cumsum(degree, 0) - degree)[owner]
    edge = steps.indptr[point][owner] + offset
    candidate = flat[states][owner] + steps.lengths[edge]
    target = tree[owner] * steps.n + steps.indices[edge]
    better = (candidate <= margin_steps) & (candidate < flat[target])
    target, candidate = target[better], candidate[better]
    flat.scatter_reduce_(0, target, candidate, reduce="amin")
    return target
