"""Times the torch forest on a saved space's Hadamards, and compares it with the space's saved forest when there is one.

    python scripts/bench_forest.py <space dir> [--k 5] [--device cuda] [--normalise]

- The Hadamards are saved before the distances and the forest, so a space still being collected can be timed.
- The comparison is the regression check for the port: the same parents and trees as the numpy forest.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from in_tfm.density_peaks import calculate_probs, forest_trees, torch_forest
from in_tfm.hit_space import FOREST, HADAMARDS, PROBABILITIES, TREES


def timed(label: str, work):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = work()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[{label}] {time.perf_counter() - start:.1f}s", flush=True)
    return result


def compare_with_saved(space: Path, forest, trees, probabilities) -> None:
    saved, saved_trees = np.load(space / FOREST), np.load(space / TREES)
    print("same distinct points:", np.array_equal(forest.distinct_of_hit, saved["distinct_of_hit"]))
    print("kth distance, max abs diff:", float(np.abs(forest.kth_distance - saved["kth_distance"]).max()))
    print("same density ranks:", np.array_equal(forest.density_rank, saved["density_rank"]))
    print("parents differing:", int((forest.parent != saved["parent"]).sum()), "of", len(forest.parent))
    print("delta, max abs diff:", float(np.abs(forest.delta - saved["delta"]).max()))
    print("neighbours differing:", int((forest.neighbours != saved["neighbours"]).any(axis=1).sum()))
    print("roots:", len(trees.roots), "saved", len(saved_trees["roots"]), "| same roots:", np.array_equal(trees.roots, saved_trees["roots"]))
    print("same tree per point:", np.array_equal(trees.tree, saved_trees["tree"]))
    saved_probabilities = np.load(space / PROBABILITIES)
    if saved_probabilities.shape == probabilities.shape:
        print("probabilities, max abs diff:", float(np.abs(probabilities - saved_probabilities).max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalise", action="store_true")
    parser.add_argument("--peak-share", type=float, default=0.8)
    parser.add_argument(
        "--skip-distance-matrix", action="store_true",
        help="the full matrix is n x n float64 in host memory, which another job on the machine may need",
    )
    args = parser.parse_args()

    hadamards = np.load(args.space / HADAMARDS)
    print(f"{hadamards.shape[0]} hits of {hadamards.shape[1]} dims on {args.device}", flush=True)
    if not args.skip_distance_matrix:
        timed("distance matrix", lambda: torch_forest.pairwise_distances(hadamards, "weighted_jaccard", args.normalise, args.device))
    forest = timed("forest", lambda: torch_forest.density_peak_forest(
        hadamards, args.k, "weighted_jaccard", args.normalise, args.device,
    ))
    trees = timed("trees", lambda: forest_trees(forest))
    probabilities = timed("probabilities", lambda: calculate_probs(forest, trees, args.peak_share))
    print(f"{len(forest.counts)} distinct points, {len(trees.roots)} trees")
    if (args.space / FOREST).exists():
        compare_with_saved(args.space, forest, trees, probabilities)
    else:
        print("no saved forest to compare with yet")
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
