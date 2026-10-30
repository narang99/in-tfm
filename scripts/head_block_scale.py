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
- Probabilities are a points x trees table, so they are skipped when that table would be too large.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import adjusted_rand_score

from head_block_forests import draw_windows, entropy_bits
from in_tfm.density_peaks import DensityPeakForest, calculate_probs, forest_trees
from in_tfm.density_peaks.ivf_forest import IvfSettings, ivf_density_peak_forest
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
    n_probe: int = 16
    peak_probe: int = 64
    n_list_neighbours: int = 32
    exact_up_to: int = 256_000
    max_probability_cells: float = 2e9
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
    n_lists = max(1, round(config.lists_per_root_n * np.sqrt(n)))
    return IvfSettings(
        n_lists=n_lists, n_probe=n_lists if exact else config.n_probe,
        peak_probe=n_lists if exact else config.peak_probe, n_list_neighbours=config.n_list_neighbours,
        seed=config.seed,
    )


def forest_summary(forest: DensityPeakForest, points: BlockPoints, n: int, config: HeadBlockScaleConfig) -> dict:
    began = time.perf_counter()
    trees = forest_trees(forest)
    tree = trees.tree[forest.distinct_of_hit]
    summary = {
        "trees": len(trees.roots), "largest_tree": float(np.bincount(tree).max() / n),
        "purity_previous": purity(tree, points.previous[:n]), "purity_current": purity(tree, points.current[:n]),
    }
    if n * len(trees.roots) <= config.max_probability_cells:
        probabilities = calculate_probs(forest, trees, config.peak_share)[forest.distinct_of_hit]
        entropy = entropy_bits(probabilities)
        summary |= {"median_entropy": float(np.median(entropy)),
                    "confident": float((entropy < config.confident_bits).mean())}
    summary["summary_seconds"] = time.perf_counter() - began
    return summary, tree


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
    summary, tree = forest_summary(forest, points, n, config)
    row = {"n": n, "exact": exact, "n_lists": settings.n_lists, "n_probe": settings.n_probe,
           "peak_probe": settings.peak_probe, "forest_seconds": seconds, **stats.model_dump(), **summary}
    return row, forest, tree


def main() -> None:
    config = HeadBlockScaleConfig()
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
