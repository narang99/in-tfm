#!/usr/bin/env python
"""Which clusters touch each other, read from the arrays `collect_hit_space.py` saved.

    python scripts/cluster_connectivity.py /content/runs/mixed4e_55/space --ks 5 10 20

- Outliers are the hits with no mutual neighbour: i and j are linked only if each is in the other's k nearest.
- Two clusters are directly connected if some mutual edge has a hit of each at its ends.
- A hit's cluster is its density-peak tree, either for every hit, or only for the confident ones (entropy below the cut).
"""

import argparse
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

MAX_ENTROPY_BITS = 0.5
UNLABELLED = -1


def nearest_neighbours(distances: Float[np.ndarray, "n n"], k: int) -> Int[np.ndarray, "n k"]:
    """The k closest other hits per row. Ties between equal distances break arbitrarily."""
    masked = np.array(distances, dtype=np.float32)
    np.fill_diagonal(masked, np.inf)
    return np.argpartition(masked, k, axis=1)[:, :k]


def mutual_edges(neighbours: Int[np.ndarray, "n k"]) -> Int[np.ndarray, "n_edges 2"]:
    n, k = neighbours.shape
    directed = csr_matrix((np.ones(n * k, dtype=bool), (np.repeat(np.arange(n), k), neighbours.ravel())), shape=(n, n))
    mutual = directed.multiply(directed.T).tocoo()
    keep = mutual.row < mutual.col
    return np.stack([mutual.row[keep], mutual.col[keep]], axis=1)


def hit_trees(space: Path) -> Int[np.ndarray, "n"]:
    forest, trees = np.load(space / "forest.npz"), np.load(space / "trees.npz")
    return trees["tree"][forest["distinct_of_hit"]]


def hit_entropy_bits(space: Path) -> Float[np.ndarray, "n"]:
    forest, probabilities = np.load(space / "forest.npz"), np.load(space / "probabilities.npy")
    p = probabilities[forest["distinct_of_hit"]]
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)


def cluster_pair_edges(edges: Int[np.ndarray, "n_edges 2"], labels: Int[np.ndarray, "n"], n_clusters: int) -> Int[np.ndarray, "c c"]:
    """Symmetric count of mutual edges between each pair of clusters. Edges to unlabelled hits are not counted."""
    a, b = labels[edges[:, 0]], labels[edges[:, 1]]
    both = (a != UNLABELLED) & (b != UNLABELLED)
    counts = coo_matrix((np.ones(both.sum(), dtype=np.int64), (a[both], b[both])), shape=(n_clusters, n_clusters)).toarray()
    return counts + counts.T - np.diag(np.diag(counts))


def unconnected_pairs(pair_edges: Int[np.ndarray, "c c"], sizes: Int[np.ndarray, "c"], min_size: int) -> list[tuple[int, int]]:
    big = np.flatnonzero(sizes >= min_size)
    return [(int(a), int(b)) for i, a in enumerate(big) for b in big[i + 1:] if pair_edges[a, b] == 0]


def summarise(name: str, k: int, edges: Int[np.ndarray, "n_edges 2"], labels: Int[np.ndarray, "n"], min_size: int) -> dict:
    n_clusters = int(labels.max()) + 1
    sizes = np.bincount(labels[labels != UNLABELLED], minlength=n_clusters)
    pair_edges = cluster_pair_edges(edges, labels, n_clusters)
    n_components, component_of = connected_components(csr_matrix(pair_edges > 0), directed=False)
    big = sizes >= min_size
    print(f"\n[{name}, k={k}] clusters of >= {min_size} hits: {int(big.sum())}")
    print(f"  cluster-graph components among them: {len(set(component_of[big].tolist()))}")
    for component in sorted(set(component_of[big].tolist())):
        members = np.flatnonzero(big & (component_of == component))
        print(f"    component {component}: clusters {members.tolist()}")
    print(f"  pairs of those clusters with no direct edge: {len(unconnected_pairs(pair_edges, sizes, min_size))} "
          f"of {int(big.sum()) * (int(big.sum()) - 1) // 2}")
    return {"sizes": sizes, "pair_edges": pair_edges}


def report_k(space: Path, k: int, distances: Float[np.ndarray, "n n"], trees: Int[np.ndarray, "n"],
             confident: Bool[np.ndarray, "n"], min_size: int) -> None:
    edges = mutual_edges(nearest_neighbours(distances, k))
    degree = np.bincount(edges.ravel(), minlength=len(trees))
    outliers = degree == 0
    print(f"\n=== k={k}: {len(edges)} mutual edges, {int(outliers.sum())} of {len(trees)} hits have none (outliers) ===")
    _, hit_component = connected_components(
        csr_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])), shape=(len(trees),) * 2), directed=False
    )
    sizes = np.bincount(hit_component[~outliers])
    print(f"  hit-level components: {int((sizes > 0).sum())}, largest {np.sort(sizes)[::-1][:5].tolist()}")
    results = {}
    for name, labels in {"all hits by tree": trees, "confident hits only": np.where(confident, trees, UNLABELLED)}.items():
        results[name] = summarise(name, k, edges, labels, min_size)
    np.savez(space / f"connectivity_k{k}.npz", edges=edges, outliers=outliers, hit_component=hit_component,
             **{f"{name.replace(' ', '_')}__{key}": value for name, r in results.items() for key, value in r.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=[5])
    parser.add_argument("--min-size", type=int, default=100, help="clusters smaller than this are left out of the summary")
    args = parser.parse_args()

    distances = np.load(args.space / "distances.npy")
    trees = hit_trees(args.space)
    confident = hit_entropy_bits(args.space) < MAX_ENTROPY_BITS
    print(f"{len(trees)} hits, {int(confident.sum())} confident (entropy < {MAX_ENTROPY_BITS} bits), {trees.max() + 1} trees")
    for k in args.ks:
        report_k(args.space, k, distances, trees, confident, args.min_size)


if __name__ == "__main__":
    main()
