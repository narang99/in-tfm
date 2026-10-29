#!/usr/bin/env python
"""Density-peak forests on single head blocks of layer 0 o_proj neurons, as the clustering sample grows.

    python scripts/head_block_forests.py --checkpoint-dir checkpoints/tiny_attn_only_2l --neurons "[0,1,2,3]"

- Neuron d's head-h block is `z_h ⊙ W_O[d, h-block]`, the part of the o_proj Hadamard row that head h writes.
- Positions are drawn from the model's own training stream, one per window, so samples are independent.
  - Each window starts at a story's BOS, as in training, and the position is uniform within it.
  - The sizes are nested prefixes of one draw, so a larger size only adds positions.
- Each tree is scored against three token labels of its hits: the previous token, the current one, and the pair.
  - Purity is the share of hits that carry their tree's most common label.
- Head types come from layer 0's attention at the sampled positions: the share on BOS, the previous token and self.
- Induction is checked on a repeated random sequence, per layer 1 head.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import normalized_mutual_info_score

from in_tfm.density_peaks import calculate_probs, density_peak_forest, forest_trees, torch_forest
from in_tfm.density_peaks.probabilities import step_length
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, attention_patterns, load_trained_attn_only
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts


class HeadBlockForestsConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l")
    split: str = "train"
    neurons: list[int] = [0, 1, 2, 3]
    heads: list[int] | None = None
    """All of layer 0's heads when unset."""
    sizes: list[int] = [1000, 2000, 4000, 8000, 16000]
    k: int = 5
    peak_share: float = 0.8
    confident_bits: float = 0.5
    seed: int = 0
    batch_size: int = 256
    device: str = "cuda"
    out: Path = Path("reports/head_block_forests/results.jsonl")


class Sample(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    z: Float[np.ndarray, "n inner"]
    current: Int[np.ndarray, "n"]
    previous: Int[np.ndarray, "n"]
    """-1 at position 0, which has no previous token."""
    position: Int[np.ndarray, "n"]
    attention_shares: Float[np.ndarray, "n head 4"]
    """Layer 0 attention on BOS, the previous token, self, and everything else."""


def draw_windows(tokens, starts, n: int, seq_len: int, rng: np.random.Generator):
    chosen = rng.choice(starts, size=n)
    windows = np.stack([tokens[s:s + seq_len] for s in chosen]).astype(np.int64)
    return windows, rng.integers(0, seq_len, size=n)


def attention_shares(pattern: Float[torch.Tensor, "batch head seq seq"], position: Int[torch.Tensor, "batch"]):
    rows = pattern[torch.arange(len(position)), :, position]
    bos = rows[..., 0]
    previous = torch.where(position[:, None] > 0, rows.gather(2, (position - 1).clamp(min=0)[:, None, None]
                                                              .expand(-1, rows.shape[1], 1))[..., 0], 0.0)
    own = rows.gather(2, position[:, None, None].expand(-1, rows.shape[1], 1))[..., 0]
    own = torch.where(position[:, None] > 0, own, 0.0)
    return torch.stack([bos, previous, own, 1 - bos - previous - own], dim=-1)


@torch.no_grad()
def sample_positions(model, tokens, starts, config: HeadBlockForestsConfig, seq_len: int) -> Sample:
    rng = np.random.default_rng(config.seed)
    windows, position = draw_windows(tokens, starts, max(config.sizes), seq_len, rng)
    o_proj = model.model.layers[0].self_attn.o_proj
    captured = {}
    handle = o_proj.register_forward_hook(lambda _m, inputs, _o: captured.__setitem__("z", inputs[0]))
    z, shares = [], []
    try:
        for start in range(0, len(windows), config.batch_size):
            ids = torch.from_numpy(windows[start:start + config.batch_size]).to(config.device)
            at = torch.from_numpy(position[start:start + config.batch_size]).to(config.device)
            pattern = attention_patterns(model, ids, 0)
            z.append(captured["z"][torch.arange(len(at)), at].cpu())
            shares.append(attention_shares(pattern, at).cpu())
    finally:
        handle.remove()
    rows = np.arange(len(windows))
    return Sample(
        z=torch.cat(z).double().numpy(), current=windows[rows, position],
        previous=np.where(position > 0, windows[rows, np.maximum(position - 1, 0)], -1), position=position,
        attention_shares=torch.cat(shares).numpy(),
    )


@torch.no_grad()
def induction_scores(model, vocab_size: int, bos: int, device: str, batch: int = 32, seed: int = 0):
    """Per layer 1 head: mean attention from the second copy to the token after the same token in the first copy.

    Also per layer 0 head: mean attention to the previous token on the same sequences."""
    length = min(50, (model.config.max_position_embeddings - 1) // 2)
    rng = np.random.default_rng(seed)
    half = rng.integers(1, vocab_size, size=(batch, length))
    ids = torch.from_numpy(np.concatenate([np.full((batch, 1), bos), half, half], axis=1)).to(device)
    second = torch.arange(length + 1, 2 * length + 1, device=device)
    layer1 = attention_patterns(model, ids, 1)[:, :, second, second - length + 1].mean(dim=(0, 2))
    every = torch.arange(1, 2 * length + 1, device=device)
    layer0 = attention_patterns(model, ids, 0)[:, :, every, every - 1].mean(dim=(0, 2))
    logits = model(input_ids=ids).logits.float()
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    return {
        "layer1_induction": layer1.tolist(), "layer0_previous": layer0.tolist(),
        "loss_first_copy": loss[:, 1:length].mean().item(), "loss_second_copy": loss[:, length + 1:].mean().item(),
    }


def purity(tree: Int[np.ndarray, "n"], labels: np.ndarray) -> float:
    total = 0
    for t in np.unique(tree):
        _, counts = np.unique(labels[tree == t], return_counts=True)
        total += counts.max()
    return total / len(tree)


def entropy_bits(probabilities: Float[np.ndarray, "n n_trees"]) -> Float[np.ndarray, "n"]:
    safe = np.where(probabilities > 0, probabilities, 1)
    return -(probabilities * np.log2(safe)).sum(axis=1)


def block_rows(sample: Sample, weight: Float[np.ndarray, "inner"], head: int, head_dim: int, n: int):
    block = slice(head * head_dim, (head + 1) * head_dim)
    return sample.z[:n, block] * weight[block]


def forest_result(rows, sample: Sample, n: int, config: HeadBlockForestsConfig) -> dict:
    began = time.time()
    forest = torch_forest.density_peak_forest(rows, config.k, "weighted_jaccard", False, config.device) \
        if config.device.startswith("cuda") else density_peak_forest(rows, config.k, "weighted_jaccard", False)
    trees = forest_trees(forest)
    probabilities = calculate_probs(forest, trees, config.peak_share)[forest.distinct_of_hit]
    seconds = time.time() - began
    tree = trees.tree[forest.distinct_of_hit]
    entropy = entropy_bits(probabilities)
    pair = sample.previous[:n] * 1_000_003 + sample.current[:n]
    labels = {"previous": sample.previous[:n], "current": sample.current[:n], "pair": pair}
    return {
        "distinct": len(forest.counts), "trees": len(trees.roots), "tau": step_length(forest),
        "median_entropy": float(np.median(entropy)), "confident": float((entropy < config.confident_bits).mean()),
        "largest_tree": float(np.bincount(tree).max() / n), "seconds": seconds,
        **{f"purity_{name}": purity(tree, values) for name, values in labels.items()},
        **{f"nmi_{name}": float(normalized_mutual_info_score(values, tree)) for name, values in labels.items()},
        "distinct_labels": {name: int(len(np.unique(values))) for name, values in labels.items()},
    }


def main() -> None:
    config = HeadBlockForestsConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    model = model.to(config.device)
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split=config.split))
    bos = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    seq_len = model.config.max_position_embeddings
    head_dim = model.config.head_dim
    heads = config.heads if config.heads is not None else list(range(model.config.num_attention_heads))
    config.out.parent.mkdir(parents=True, exist_ok=True)

    scores = induction_scores(model, model.config.vocab_size, bos, config.device)
    print(json.dumps(scores, indent=1))
    sample = sample_positions(model, tokens, story_starts(tokens, bos, seq_len), config, seq_len)
    head_shares = sample.attention_shares[sample.position > 0].mean(axis=0)
    for head in heads:
        print(f"layer 0 head {head}: bos {head_shares[head, 0]:.2f} previous {head_shares[head, 1]:.2f} "
              f"self {head_shares[head, 2]:.2f} other {head_shares[head, 3]:.2f}")
    weights = model.model.layers[0].self_attn.o_proj.weight.detach().double().cpu().numpy()
    with config.out.open("w") as out:
        out.write(json.dumps({"kind": "model", **scores, "head_shares": head_shares.tolist()}) + "\n")
        for neuron in config.neurons:
            for head in heads:
                for n in config.sizes:
                    result = forest_result(block_rows(sample, weights[neuron], head, head_dim, n), sample, n, config)
                    row = {"kind": "forest", "neuron": neuron, "head": head, "n": n, **result}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"neuron {neuron} head {head} n {n:6d}: trees {result['trees']:4d} "
                          f"confident {result['confident']:5.1%} entropy {result['median_entropy']:.2f} "
                          f"purity prev {result['purity_previous']:.2f} cur {result['purity_current']:.2f} "
                          f"pair {result['purity_pair']:.2f} largest {result['largest_tree']:.2f} "
                          f"{result['seconds']:.1f}s")
    print("done")


if __name__ == "__main__":
    main()
