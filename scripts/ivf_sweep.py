#!/usr/bin/env python
"""Search settings for the IVF forest, each scored against one exact forest on the same points.

    python scripts/ivf_sweep.py --head 0 --size 1000000 --variants '[{"probe_by": "cell", "cell_size": 1000, "n_probe": 64}]'

- The points and the exact forest are built once, then each variant reuses them.
- A variant sets any `HeadBlockScaleConfig` field that `settings_for` reads.
- No entropy: the forest's agreement with the exact one is what the settings change.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

from head_block_forests import draw_windows
from head_block_scale import HeadBlockScaleConfig, agreement, block_points, run_forest
from in_tfm.density_peaks import ivf_forest
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts


class IvfSweepConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    neuron: int = 0
    head: int = 0
    size: int = 1_000_000
    variants: list[dict] = [{}]
    block_elements: int = 2**28
    out: Path = Path("reports/ivf_sweep/results.jsonl")


def main() -> None:
    sweep = IvfSweepConfig()
    ivf_forest.BLOCK_ELEMENTS = sweep.block_elements
    base = HeadBlockScaleConfig(_cli_parse_args=False, checkpoint_dir=sweep.checkpoint_dir, neuron=sweep.neuron,
                                head=sweep.head, sizes=[sweep.size], entropy=False, exact_on_device=True,
                                out=sweep.out, block_elements=sweep.block_elements)
    model, tokenizer = load_trained_attn_only(base.checkpoint_dir)
    model = model.to("cuda").eval()
    tokens = np.load(base.checkpoint_dir / TOKENS_FILE.format(split="train"), mmap_mode="r")
    seq_len = model.config.max_position_embeddings
    starts = story_starts(tokens, tokenizer.convert_tokens_to_ids(BOS_TOKEN), seq_len)
    windows, _ = draw_windows(tokens, starts, -(-sweep.size // seq_len), seq_len, np.random.default_rng(base.seed))
    points = block_points(model, windows, sweep.neuron, sweep.head, base)
    exact_row, exact_forest, exact_tree = run_forest(points, sweep.size, base, exact=True)
    print("exact", json.dumps({k: exact_row[k] for k in ("forest_seconds", "trees")}), flush=True)
    sweep.out.parent.mkdir(parents=True, exist_ok=True)
    with sweep.out.open("a") as out:
        for variant in sweep.variants:
            config = base.model_copy(update=variant)
            row, forest, tree = run_forest(points, sweep.size, config, exact=False)
            row |= agreement(forest, exact_forest, tree, exact_tree) | {"variant": variant, "exact_trees": exact_row["trees"]}
            out.write(json.dumps({"neuron": sweep.neuron, "head": sweep.head, **row}) + "\n")
            out.flush()
            keys = ("forest_seconds", "seconds", "trees", "neighbour_recall", "same_parent", "tree_ari", "local_peaks")
            print("variant", json.dumps(variant), json.dumps({k: row[k] for k in keys}), flush=True)
            del forest
            torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
