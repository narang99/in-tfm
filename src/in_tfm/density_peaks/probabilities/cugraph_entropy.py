"""`geodesic_entropy` on the GPU, with cuGraph's single-source shortest paths.

- It computes what `vectorized.geodesic_entropy` computes, with float32 lengths.
  - The scipy version runs one Dijkstra per tree on one CPU core, which took 29 minutes at 4M points.
- cuGraph's `sssp` takes one source, and a tree's peak has many points.
  - So each tree gets one extra vertex, joined to its peak points by edges of the smallest float32 length.
  - One more extra vertex joins every peak point. Its search gives each point's distance to its nearest peak.
- `hybrid_geodesic_entropy` splits the trees between the GPU and forked CPU workers, which run at the same time.
  - On an A100 VM, head 0 at 4M took 366 s on the GPU alone and 290 s on 12 CPU workers.
- Only for CUDA, with `pylibcugraph` and `cupy` installed, as on Colab's GPU runtimes.
  - The tests cannot run it, so `scripts/head_block_scale.py` compares it with the scipy version up to a size.
"""

import cupy as cp
import numpy as np
import pylibcugraph as cugraph
import scipy.sparse as sp
from jaxtyping import Float, Int

from ..forest import DensityPeakForest, ForestTrees
from .vectorized import _past_nearest_peak, _start_tree_sums, knn_graph, peak_cores, step_length

SOURCE_EDGE = float(np.finfo(np.float32).tiny)
"""cuGraph keeps a zero-length edge, but the scipy version cannot, so both use the smallest length."""
NO_CUTOFF = float(np.finfo(np.float32).max)


def gpu_geodesic_entropy(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float, margin_steps: float
) -> Float[np.ndarray, "n_distinct"]:
    handle = cugraph.ResourceHandle()
    steps, cores = steps_past_nearest_peak(handle, forest, trees, peak_share)
    total_weight, weighted_steps = gpu_tree_sums(handle, steps, cores, margin_steps)
    return entropy_bits(total_weight, weighted_steps)


def hybrid_geodesic_entropy(
    forest: DensityPeakForest, trees: ForestTrees, peak_share: float, margin_steps: float, workers: int,
    gpu_share: float,
) -> Float[np.ndarray, "n_distinct"]:
    """`gpu_share` of the trees go to the GPU, dealt out in turn so both sides get large and small trees."""
    handle = cugraph.ResourceHandle()
    steps, cores = steps_past_nearest_peak(handle, forest, trees, peak_share)
    on_gpu = np.floor((np.arange(len(cores)) + 1) * gpu_share) > np.floor(np.arange(len(cores)) * gpu_share)
    cpu_sums = _start_tree_sums(steps, [c for c, g in zip(cores, on_gpu) if not g], margin_steps, workers)
    gpu_total, gpu_weighted = gpu_tree_sums(handle, steps, [c for c, g in zip(cores, on_gpu) if g], margin_steps)
    cpu_total, cpu_weighted = cpu_sums()
    return entropy_bits(gpu_total + cpu_total, gpu_weighted + cpu_weighted)


def steps_past_nearest_peak(
    handle: cugraph.ResourceHandle, forest: DensityPeakForest, trees: ForestTrees, peak_share: float
) -> tuple[sp.csr_matrix, list[np.ndarray]]:
    """The reweighted graph of `vectorized._past_nearest_peak`, with the nearest-peak distances from the GPU."""
    graph = knn_graph(forest)
    cores = peak_cores(forest, trees, peak_share)
    tau = step_length(forest)
    n = graph.shape[0]
    all_peaks = to_gpu_graph(handle, with_sources(graph / tau, [np.concatenate(cores)]))
    nearest = cp.asnumpy(shortest_from(handle, all_peaks, n, NO_CUTOFF, n + 1)[:n]).astype(np.float64)
    return _past_nearest_peak(graph / tau, nearest), cores


def gpu_tree_sums(
    handle: cugraph.ResourceHandle, steps: sp.csr_matrix, cores: list[np.ndarray], margin_steps: float
) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    n = steps.shape[0]
    graph = to_gpu_graph(handle, with_sources(steps, cores))
    total_weight = cp.zeros(n, dtype=cp.float64)
    weighted_steps = cp.zeros(n, dtype=cp.float64)
    for tree in range(len(cores)):
        past = shortest_from(handle, graph, n + tree, margin_steps, n + len(cores))[:n]
        reached = cp.flatnonzero(cp.isfinite(past))
        past = past[reached].astype(cp.float64)
        weight = cp.exp(-past)
        total_weight[reached] += weight
        weighted_steps[reached] += weight * past
    return cp.asnumpy(total_weight), cp.asnumpy(weighted_steps)


def entropy_bits(total_weight: Float[np.ndarray, "n"], weighted_steps: Float[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
    nats = np.zeros(len(total_weight))
    reachable = total_weight > 0
    nats[reachable] = np.log(total_weight[reachable]) + weighted_steps[reachable] / total_weight[reachable]
    return nats / np.log(2)


def with_sources(graph: sp.csr_matrix, sources: list[Int[np.ndarray, "n_peak"]]) -> sp.coo_matrix:
    """The graph with one extra vertex per source set, numbered after the graph's own, with edges out to its points."""
    n = graph.shape[0]
    coo = graph.tocoo()
    source_ids = np.repeat(np.arange(n, n + len(sources)), [len(s) for s in sources])
    targets = np.concatenate(sources) if sources else np.zeros(0, dtype=int)
    size = n + len(sources)
    return sp.coo_matrix(
        (np.concatenate([coo.data, np.full(len(targets), SOURCE_EDGE)]),
         (np.concatenate([coo.row, source_ids]), np.concatenate([coo.col, targets]))),
        shape=(size, size),
    )


def to_gpu_graph(handle: cugraph.ResourceHandle, edges: sp.coo_matrix) -> cugraph.SGGraph:
    return cugraph.SGGraph(
        handle, cugraph.GraphProperties(is_symmetric=False, is_multigraph=False),
        cp.asarray(edges.row, dtype=cp.int32), cp.asarray(edges.col, dtype=cp.int32),
        cp.asarray(edges.data, dtype=cp.float32),
        store_transposed=False, renumber=False, do_expensive_check=False,
        vertices_array=cp.arange(edges.shape[0], dtype=cp.int32),
    )


def shortest_from(
    handle: cugraph.ResourceHandle, graph: cugraph.SGGraph, source: int, cutoff: float, n_vertices: int
) -> cp.ndarray:
    """Inf past the cutoff and out of reach. cuGraph marks those with the largest float instead."""
    vertices, distances, _ = cugraph.sssp(handle, graph, source, cutoff, True, False)
    out = cp.full(n_vertices, cp.inf, dtype=cp.float32)
    within = distances <= cutoff
    out[vertices[within]] = distances[within]
    return out
