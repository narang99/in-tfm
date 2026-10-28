#!/usr/bin/env python
"""Does a layer-0 v neuron of the attn-only model group tokens into dense clusters, or is each token its own meaning?

- Before attention mixes positions, `v = W_V W_E[t]`, so a v neuron's hit depends only on the token.
- So the whole vocabulary is clustered directly, each token once, as its Hadamard product `W_E[t] * W_V[neuron]`.
  - A corpus would only add copies, and copies make frequent tokens look dense.
  - Keeping only frequent tokens would bias the density too, so no token is left out.
- Density peaks with weighted Jaccard and no normalisation, on the GPU forest.

    python scripts/v_vocab_clusters.py --neuron 100 --device cuda
"""

import argparse
import random
import time
from collections import Counter

import numpy as np
from jaxtyping import Float, Int

from in_tfm.density_peaks import (
    calculate_probs,
    forest_trees,
    log_density,
    torch_forest,
)
from in_tfm.models.attn_only_2l import load_attn_only_2l

MAX_ENTROPY = 0.5
EXAMPLES_PER_CLUSTER = 16
CLUSTERS_SHOWN = 15


def entropy_bits(probs: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n"]:
    safe = np.where(probs > 0, probs, 1.0)
    return -(np.where(probs > 0, probs * np.log2(safe), 0.0)).sum(axis=1)


def show_clusters(
    tree: Int[np.ndarray, "n"], confident: Int[np.ndarray, "n"], activation: Float[np.ndarray, "n"], words: list[str],
) -> None:
    """Largest trees first, with their size, how many members are confident, and random member tokens."""
    rng = np.random.default_rng(0)
    for t, size in Counter(tree.tolist()).most_common(CLUSTERS_SHOWN):
        members = np.flatnonzero(tree == t)
        sure = int((confident[members] == t).sum())
        picked = rng.choice(members, min(EXAMPLES_PER_CLUSTER, len(members)), replace=False)
        acts = activation[members]
        print(f"  c{t} n={size} confident={sure} act {acts.min():+.2f}..{acts.max():+.2f}: {[words[i] for i in picked]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neuron", type=int, default=None, help="random when unset")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="where to save each token's tree and probabilities, as .npz")
    args = parser.parse_args()
    neuron = args.neuron if args.neuron is not None else random.Random().randrange(768)

    model, tokenizer = load_attn_only_2l()
    W_E = model.model.embed_tokens.weight[: len(tokenizer)].detach().double().numpy()
    W_V = model.model.layers[0].self_attn.v_proj.weight.detach().double().numpy()
    words = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    hadamard = W_E * W_V[neuron]
    activation = hadamard.sum(axis=1)
    print(f"layer 0 v_proj neuron {neuron} (head {neuron // 64}, dim {neuron % 64}), all {len(words)} tokens, "
          f"activation {activation.min():+.2f} to {activation.max():+.2f}")

    start = time.perf_counter()
    forest = torch_forest.density_peak_forest(hadamard, args.k, "weighted_jaccard", False, args.device)
    trees = forest_trees(forest)
    print(f"[forest] {time.perf_counter() - start:.1f}s, {len(forest.counts)} distinct of {len(words)}, "
          f"{len(trees.roots)} trees, cuts rho {trees.cuts.rho:.3f} delta {trees.cuts.delta:.3f}")

    start = time.perf_counter()
    probs = calculate_probs(forest, trees, 0.8)[forest.distinct_of_hit]
    print(f"[probabilities] {time.perf_counter() - start:.1f}s")
    entropy = entropy_bits(probs)
    confident = np.where(entropy < MAX_ENTROPY, probs.argmax(axis=1), -1)
    tree = trees.tree[forest.distinct_of_hit]
    rho = log_density(forest.kth_distance)[forest.distinct_of_hit]
    print(f"kth-neighbour distance median {np.median(forest.kth_distance):.3f}, "
          f"10th percentile {np.quantile(forest.kth_distance, 0.1):.3f}")
    print(f"median entropy {np.median(entropy):.2f} bits, {np.mean(entropy < MAX_ENTROPY):.0%} under {MAX_ENTROPY}, "
          f"{np.mean(entropy < 0.01):.0%} under 0.01")
    print(f"tokens above the rho cut: {np.mean(rho >= trees.cuts.rho):.0%}")
    show_clusters(tree, confident, activation, words)
    np.savez(args.out or f"v_vocab_neuron_{neuron}.npz", tree=tree, probs=probs.astype(np.float32), rho=rho,
             activation=activation, entropy=entropy)
    print("[total] done")


if __name__ == "__main__":
    main()
