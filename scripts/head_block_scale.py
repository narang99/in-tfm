#!/usr/bin/env python
"""Forests on one head block of one layer 0 o_proj neuron, at sizes an all-pairs forest cannot reach.

    python scripts/head_block_scale.py --neuron 0 --head 2 --sizes '[256000,1000000,4000000]'

- The forest is `ivf_density_peak_forest`: approximate neighbours from an IVF index, and copies not merged.
- Every position of a window is a point, so a size of n reads n / seq_len windows.
  - The sizes are nested by window, so a larger size only adds windows.
  - `head_block_forests.py` took one position per window instead. That keeps points independent but costs a forward
    pass per point, which millions of points cannot afford.
- At sizes up to `exact_up_to`, the forest is also built with every list probed, which compares every pair.
  - Comparing the two measures what the approximate search loses.
- Entropy over trees comes from `geodesic_entropy`, which needs no points x trees table.
  - Up to `compare_dense_up_to` it is checked against the entropy of the full table.
  - Each run saves every point's entropy and tree, so later plots and reports need no new run.
  - `entropy_backend="cugraph"` computes it on the GPU. Up to `compare_dense_up_to` it is also checked against scipy.
"""

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import adjusted_rand_score

from head_block_forests import draw_windows, entropy_bits
from in_tfm.density_peaks import DensityPeakForest, ForestTrees, calculate_probs, forest_trees
from in_tfm.density_peaks import ivf_forest
from in_tfm.density_peaks.ivf_forest import IvfSettings, ivf_density_peak_forest
from in_tfm.density_peaks.probabilities import geodesic_entropy
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts


class HeadBlockScaleConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    split: str = "train"
    neuron: int = 0
    head: int = 2
    sizes: list[int] = [256_000, 1_000_000, 4_000_000]
    k: int = 5
    peak_share: float = 0.8
    confident_bits: float = 0.5
    lists_per_root_n: float = 2.0
    """n_lists is this times the square root of n, so a list holds about sqrt(n) / this points."""
    cell_size: int | None = None
    """When set, n_lists is n / cell_size instead, so the search grows with n rather than n^1.5."""
    probe_by: Literal["point", "cell", "union"] = "point"
    fused: bool = True
    """On CUDA, the Triton kernel that does the search's distances and top-k together."""
    block_elements: int = 2**26
    """Distances in one block. An A100 takes 2^28 (1 GB of float32)."""
    n_probe: int = 16
    peak_probe: int = 64
    n_list_neighbours: int = 32
    exact_up_to: int = 256_000
    margin_steps: float = 10.0
    """Trees this many steps of tau past a point's nearest peak are left out of its entropy.

    - Head 0 at 256k, against the dense table: 10 steps are off by 0.016 bits on average, 8 by 0.11, 6 by 0.45."""
    compare_dense_up_to: int = 256_000
    confident_thresholds: list[float] = [0.1, 0.5, 1.0, 2.0]
    entropy_quantiles: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    histogram_bin_bits: float = 0.25
    entropy: bool = True
    entropy_backend: Literal["scipy", "cugraph", "hybrid"] = "scipy"
    entropy_workers: int = 1
    gpu_share: float = 0.45
    """With the hybrid backend, the share of trees on the GPU. The rest go to the CPU workers."""
    """Processes for the scipy entropy. Each tree's search is independent."""
    exact_on_device: bool = False
    seed: int = 0
    batch_size: int = 256
    device: str = "cuda"
    out: Path = Path("reports/head_block_scale/results.jsonl")


class BlockPoints(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    rows: Float[np.ndarray, "n head_dim"]
    current: Int[np.ndarray, "n"]
    previous: Int[np.ndarray, "n"]
    """-1 at position 0."""


@torch.no_grad()
def block_points(model, windows: Int[np.ndarray, "w seq"], neuron: int, head: int, config: HeadBlockScaleConfig):
    head_dim = model.config.head_dim
    block = slice(head * head_dim, (head + 1) * head_dim)
    o_proj = model.model.layers[0].self_attn.o_proj
    weight = o_proj.weight[neuron, block].float()
    captured = {}
    handle = o_proj.register_forward_hook(lambda _m, inputs, _o: captured.__setitem__("z", inputs[0]))
    rows = []
    try:
        for start in range(0, len(windows), config.batch_size):
            model(input_ids=torch.from_numpy(windows[start:start + config.batch_size]).to(config.device))
            rows.append((captured["z"][..., block].float() * weight).reshape(-1, head_dim).cpu())
    finally:
        handle.remove()
    previous = np.concatenate([np.full((len(windows), 1), -1), windows[:, :-1]], axis=1)
    return BlockPoints(rows=torch.cat(rows).numpy(), current=windows.reshape(-1), previous=previous.reshape(-1))


def purity(tree: Int[np.ndarray, "n"], labels: Int[np.ndarray, "n"]) -> float:
    """The share of points that carry their tree's most common label, without a pass per tree."""
    _, pair_counts = np.unique(np.stack([tree, labels]), axis=1, return_counts=True)
    pairs, _ = np.unique(np.stack([tree, labels]), axis=1, return_inverse=True)
    best = np.zeros(tree.max() + 1, dtype=np.int64)
    np.maximum.at(best, pairs[0], pair_counts)
    return float(best.sum() / len(tree))


def settings_for(n: int, config: HeadBlockScaleConfig, exact: bool) -> IvfSettings:
    by_size = -(-n // config.cell_size) if config.cell_size else round(config.lists_per_root_n * np.sqrt(n))
    n_lists = max(1, by_size)
    return IvfSettings(
        n_lists=n_lists, n_probe=n_lists if exact else config.n_probe,
        peak_probe=n_lists if exact else config.peak_probe, n_list_neighbours=config.n_list_neighbours,
        seed=config.seed, exact_on_device=config.exact_on_device, probe_by=config.probe_by, fused=config.fused,
    )


def entropy_distribution(entropy: Float[np.ndarray, "n"], n_trees: int, config: HeadBlockScaleConfig) -> dict:
    """Bits, and bits over log2(trees), since more trees allow more entropy."""
    normalised = entropy / max(np.log2(n_trees), 1e-12)
    edges = np.arange(0, np.log2(n_trees) + 2 * config.histogram_bin_bits, config.histogram_bin_bits)
    return {
        "median_entropy": float(np.median(entropy)),
        "confident": float((entropy < config.confident_bits).mean()),
        "confident_at": {str(t): float((entropy < t).mean()) for t in config.confident_thresholds},
        "entropy_quantiles": dict(zip(map(str, config.entropy_quantiles),
                                      np.quantile(entropy, config.entropy_quantiles).tolist())),
        "normalised_quantiles": dict(zip(map(str, config.entropy_quantiles),
                                         np.quantile(normalised, config.entropy_quantiles).tolist())),
        "histogram_bin_bits": config.histogram_bin_bits, "histogram": np.histogram(entropy, edges)[0].tolist(),
    }


def tree_entropy(forest: DensityPeakForest, trees: ForestTrees, config: HeadBlockScaleConfig) -> Float[np.ndarray, "n"]:
    if config.entropy_backend != "scipy":
        torch.cuda.empty_cache()
    if config.entropy_backend == "cugraph":
        from in_tfm.density_peaks.probabilities.cugraph_entropy import gpu_geodesic_entropy

        return gpu_geodesic_entropy(forest, trees, config.peak_share, config.margin_steps)
    if config.entropy_backend == "hybrid":
        from in_tfm.density_peaks.probabilities.cugraph_entropy import hybrid_geodesic_entropy

        return hybrid_geodesic_entropy(forest, trees, config.peak_share, config.margin_steps, config.entropy_workers,
                                       config.gpu_share)
    return geodesic_entropy(forest, trees, config.peak_share, config.margin_steps, config.entropy_workers)


def scipy_check(forest: DensityPeakForest, trees: ForestTrees, entropy, config: HeadBlockScaleConfig) -> dict:
    began = time.perf_counter()
    reference = geodesic_entropy(forest, trees, config.peak_share, config.margin_steps,
                                 config.entropy_workers)[forest.distinct_of_hit]
    return {"scipy_seconds": time.perf_counter() - began,
            "scipy_max_difference": float(np.abs(reference - entropy).max()),
            "scipy_mean_difference": float((reference - entropy).mean())}


def dense_check(forest: DensityPeakForest, trees: ForestTrees, entropy, config: HeadBlockScaleConfig) -> dict:
    began = time.perf_counter()
    dense = entropy_bits(calculate_probs(forest, trees, config.peak_share)[forest.distinct_of_hit])
    return {"dense_seconds": time.perf_counter() - began, "dense_max_difference": float(np.abs(dense - entropy).max()),
            "dense_mean_difference": float((dense - entropy).mean()),
            "dense_median_entropy": float(np.median(dense)),
            "dense_confident": float((dense < config.confident_bits).mean())}


def forest_summary(forest: DensityPeakForest, points: BlockPoints, n: int, config: HeadBlockScaleConfig):
    began = time.perf_counter()
    trees = forest_trees(forest)
    tree = trees.tree[forest.distinct_of_hit]
    summary = {
        "trees": len(trees.roots), "largest_tree": float(np.bincount(tree).max() / n),
        "purity_previous": purity(tree, points.previous[:n]), "purity_current": purity(tree, points.current[:n]),
    }
    if not config.entropy:
        summary["summary_seconds"] = time.perf_counter() - began
        return summary, tree, np.zeros(n)
    entropy_began = time.perf_counter()
    entropy = tree_entropy(forest, trees, config)[forest.distinct_of_hit]
    summary["entropy_seconds"] = time.perf_counter() - entropy_began
    summary |= entropy_distribution(entropy, len(trees.roots), config)
    summary["summary_seconds"] = time.perf_counter() - began
    if n <= config.compare_dense_up_to:
        summary |= dense_check(forest, trees, entropy, config)
        if config.entropy_backend != "scipy":
            summary |= scipy_check(forest, trees, entropy, config)
    return summary, tree, entropy


def agreement(approximate: DensityPeakForest, exact: DensityPeakForest, tree_a, tree_e) -> dict:
    found = (approximate.neighbours[:, :, None] == exact.neighbours[:, None, :]).any(axis=1)
    relative = np.abs(approximate.kth_distance - exact.kth_distance) / np.maximum(exact.kth_distance, 1e-12)
    return {
        "neighbour_recall": float(found.mean()), "same_parent": float((approximate.parent == exact.parent).mean()),
        "kth_relative_error_median": float(np.median(relative)), "tree_ari": float(adjusted_rand_score(tree_e, tree_a)),
    }


def run_forest(points: BlockPoints, n: int, config: HeadBlockScaleConfig, exact: bool):
    settings = settings_for(n, config, exact)
    began = time.perf_counter()
    forest, stats = ivf_density_peak_forest(points.rows[:n], settings, config.k, "weighted_jaccard", False,
                                            config.device)
    seconds = time.perf_counter() - began
    summary, tree, entropy = forest_summary(forest, points, n, config)
    row = {"n": n, "exact": exact, "n_lists": settings.n_lists, "n_probe": settings.n_probe,
           "peak_probe": settings.peak_probe, "margin_steps": config.margin_steps,
           "probe_by": settings.probe_by, "cell_size": config.cell_size,
           "entropy_backend": config.entropy_backend, "forest_seconds": seconds,
           **stats.model_dump(), **summary}
    if config.entropy:
        save_point_arrays(entropy, tree, n, exact, config)
    return row, forest, tree


def save_point_arrays(
    entropy: Float[np.ndarray, "n"], tree: Int[np.ndarray, "n"], n: int, exact: bool, config: HeadBlockScaleConfig
) -> None:
    """Points are in window order, so the windows that `draw_windows` gives for the seed say which token each is."""
    kind = "exact" if exact else f"probe{config.n_probe}"
    name = f"neuron{config.neuron}_head{config.head}_n{n}_{kind}"
    np.save(config.out.parent / f"entropy_{name}.npy", entropy.astype(np.float32))
    np.save(config.out.parent / f"tree_{name}.npy", tree.astype(np.int32))


def main() -> None:
    config = HeadBlockScaleConfig()
    ivf_forest.BLOCK_ELEMENTS = config.block_elements
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    model = model.to(config.device).eval()
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split=config.split), mmap_mode="r")
    bos = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    seq_len = model.config.max_position_embeddings
    n_windows = -(-max(config.sizes) // seq_len)
    windows, _ = draw_windows(tokens, story_starts(tokens, bos, seq_len), n_windows, seq_len,
                              np.random.default_rng(config.seed))
    began = time.perf_counter()
    points = block_points(model, windows, config.neuron, config.head, config)
    print(f"{len(points.rows):,} points from {n_windows:,} windows in {time.perf_counter() - began:.0f}s", flush=True)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("a") as out:
        for n in config.sizes:
            row, forest, tree = run_forest(points, n, config, exact=False)
            if n <= config.exact_up_to:
                exact_row, exact_forest, exact_tree = run_forest(points, n, config, exact=True)
                out.write(json.dumps({"neuron": config.neuron, "head": config.head, **exact_row}) + "\n")
                print("exact", json.dumps(exact_row), flush=True)
                row |= agreement(forest, exact_forest, tree, exact_tree)
            out.write(json.dumps({"neuron": config.neuron, "head": config.head, **row}) + "\n")
            out.flush()
            print("ivf", json.dumps(row), flush=True)
            del forest
            torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
