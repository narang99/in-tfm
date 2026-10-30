#!/usr/bin/env python
"""Sorted output of one head block of one layer 0 o_proj neuron, with the hit selection's elbow marked.

    python scripts/head_block_elbow.py --checkpoint-dir checkpoints/tiny_attn_only_2l_v4 --neuron 0 --head 2

- The block's output is the sum over its 32 dims of `z_h ⊙ W_O[d, h-block]`, which is what head h adds to neuron d.
- Positions are drawn as in `head_block_forests.py`, with the same seed, so the plot reads the forests' sample.
- Each tail gets its own plot, since the elbow only reads positive values.
"""

from pathlib import Path

import numpy as np
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

from head_block_forests import HeadBlockForestsConfig, block_rows, sample_positions
from in_tfm.hit_selection.strategies.elbow import find_elbow_index_in_sorted_data
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts
from in_tfm.viz import save_elbow_plot


class HeadBlockElbowConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    split: str = "train"
    neuron: int = 0
    head: int = 2
    n: int = 256_000
    seed: int = 0
    device: str = "cuda"
    out_dir: Path = Path("reports/head_block_forests/elbow")


def tails(output: np.ndarray) -> dict[str, np.ndarray]:
    return {"positive": np.sort(output[output > 0]), "negative": np.sort(-output[output < 0])}


def main() -> None:
    config = HeadBlockElbowConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    model = model.to(config.device)
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split=config.split), mmap_mode="r")
    bos = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    seq_len = model.config.max_position_embeddings
    sampling = HeadBlockForestsConfig(_cli_parse_args=False, sizes=[config.n], seed=config.seed, device=config.device)
    sample = sample_positions(model, tokens, story_starts(tokens, bos, seq_len), sampling, seq_len)
    weights = model.model.layers[0].self_attn.o_proj.weight.detach().double().cpu().numpy()
    output = block_rows(sample, weights[config.neuron], config.head, model.config.head_dim, config.n).sum(axis=1)
    print(f"all {len(output)}: mean {output.mean():.4f} std {output.std():.4f} "
          f"min {output.min():.4f} max {output.max():.4f}")

    stem = f"neuron_{config.neuron}_head_{config.head}"
    config.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.out_dir / f"{stem}_output.npy", output)
    for polarity, ascending in tails(output).items():
        elbow_idx = find_elbow_index_in_sorted_data(torch.from_numpy(ascending))
        path = config.out_dir / f"{stem}_{polarity}_elbow.png"
        save_elbow_plot(ascending, elbow_idx, path)
        print(f"{polarity}: {len(ascending)} positions, elbow at {elbow_idx} value {ascending[elbow_idx]:.4f}, "
              f"{len(ascending) - elbow_idx} above, max {ascending[-1]:.4f} -> {path}")
    print("done")


if __name__ == "__main__":
    main()
