#!/usr/bin/env python
"""A neuron report for one TinyStories-1M MLP write, on the tokens where the stream-writers walk keeps it.

- Each story is walked back from every logit, as `ablate_unpicked_neurons.py --mode stream-writers` does.
  - The walk splits each stream dim into every earlier write, and keeps each sign's top `--share`.
  - Scores start no walks, since the QK circuit is studied separately.
  - Nothing is ablated. The walk only decides which tokens become hits.
- A token is a hit when the walk keeps that write there. Each tail keeps the hits whose activation has its sign.
- Everything after the selection is the standard report, from `neuron_run.report_neuron`.

The walk is the slow part, so its picks are saved to `<out_dir>/walk_picks.pt` and reused.

    python scripts/writer_neuron_report.py --config configs/tinystories_1m_l6_mlp_write.yaml --n-stories 80
"""

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Bool, Int
from transformers import GPTNeoForCausalLM, PreTrainedTokenizerBase

from in_tfm.activation_ranges import sample_noise
from in_tfm.hit_selection import HitSelection
from in_tfm.hit_selection.picks import RandomPick
from in_tfm.hit_selection.selection import (
    TailSelection,
    merged_selection,
    positive_sorted_activations,
    tail_budgets,
    tail_column,
    tails_of,
)
from in_tfm.layers import submodule_getter
from in_tfm.models.gpt_neo import GptNeoAdapter
from in_tfm.neuron_capture import ScanResult
from in_tfm.neuron_run import prepare_run, report_neuron, timed
from in_tfm.run_config import RunConfig, load_run_config
from in_tfm.useful_neurons.gpt_neo.trace import trace_gpt_neo
from in_tfm.useful_neurons.gpt_neo.writer_picks import mlp_write_picks

WALK_PICKS = "walk_picks.pt"
LAYER_OF_NAME = re.compile(r"^h\.(\d+)\.mlp\.c_proj$")
_worker_model: GPTNeoForCausalLM | None = None

StoryPicks = Bool[np.ndarray, "seq n_neurons"]


def parse_walk_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--share", type=float, default=0.9)
    parser.add_argument("--n-stories", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_known_args(argv)


def write_layer(layer_name: str) -> int:
    match = LAYER_OF_NAME.match(layer_name)
    if match is None:
        raise ValueError(f"{layer_name!r} is not an MLP write, such as h.6.mlp.c_proj")
    return int(match.group(1))


def tinystories(n: int) -> list[str]:
    return load_dataset("roneneldan/TinyStories", split="validation")["text"][:n]


def story_tokens(
    tokenizer: PreTrainedTokenizerBase, story: str, max_length: int
) -> Int[torch.Tensor, " seq"]:
    """Truncated as `TextSource` truncates, without its padding."""
    ids = tokenizer(story, truncation=True, max_length=max_length)["input_ids"]
    return torch.tensor(ids)


def start_worker(model_name: str) -> None:
    global _worker_model
    torch.set_num_threads(1)
    _worker_model = GPTNeoForCausalLM.from_pretrained(model_name).eval()


def walk_story(
    task: tuple[Int[torch.Tensor, " seq"], int, list[int], float],
) -> StoryPicks:
    tokens, layer, dims, share = task
    assert _worker_model is not None
    picks = mlp_write_picks(trace_gpt_neo(_worker_model, tokens), layer, share)
    return picks[:, dims].numpy()


def walk_stories(
    config: RunConfig,
    tokenizer: PreTrainedTokenizerBase,
    stories: list[str],
    share: float,
    workers: int,
) -> list[StoryPicks]:
    layer = write_layer(config.layer_name)
    tasks = [
        (
            story_tokens(tokenizer, s, config.text.max_length),
            layer,
            config.neuron_idxs,
            share,
        )
        for s in stories
    ]
    picks = []
    with Pool(
        workers, initializer=start_worker, initargs=(config.model.value,)
    ) as pool:
        for i, story_picks in enumerate(pool.imap(walk_story, tasks)):
            picks.append(story_picks)
            if (i + 1) % 10 == 0:
                print(f"walked {i + 1}/{len(tasks)} stories", flush=True)
    return picks


def saved_or_walked(
    config: RunConfig,
    tokenizer: PreTrainedTokenizerBase,
    stories: list[str],
    share: float,
    workers: int,
) -> list[StoryPicks]:
    path = Path(config.out_dir) / WALK_PICKS
    settings = dict(
        layer_name=config.layer_name,
        neurons=config.neuron_idxs,
        share=share,
        max_length=config.text.max_length,
        n_stories=len(stories),
    )
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved["settings"] == settings:
            print(f"walk picks <- {path}")
            return saved["picks"]
    with timed(f"walk {len(stories)} stories"):
        picks = walk_stories(config, tokenizer, stories, share, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"settings": settings, "picks": picks}, path)
    return picks


def picked_columns(
    picks: list[StoryPicks],
    column: int,
    valid_mask: Bool[torch.Tensor, "batch seq"],
    right_padded: bool,
) -> Bool[torch.Tensor, "batch seq"]:
    """Story picks placed in the scan's fixed-width columns."""
    width = valid_mask.shape[1]
    placed = torch.zeros_like(valid_mask)
    for row, story in enumerate(picks):
        start = 0 if right_padded else width - len(story)
        placed[row, start : start + len(story)] = torch.from_numpy(story[:, column])
    return placed & valid_mask


def walk_selection(
    scan: ScanResult,
    neuron_idx: int,
    picked: Bool[torch.Tensor, "batch seq"],
    config: RunConfig,
) -> HitSelection:
    """`RandomPick` with floor 0: every picked token of the tail's sign is a candidate, capped at `max_hits`."""
    rng = np.random.default_rng(config.seed)
    columns = {t: tail_column(scan, neuron_idx, t) for t in tails_of(config.polarity)}
    candidates = {
        t: (picked & (column[..., 0] > 0)).nonzero().numpy()
        for t, column in columns.items()
    }
    firing = [t for t in columns if len(candidates[t])]
    budgets = tail_budgets([len(candidates[t]) for t in firing], config.hits.max_hits)
    tails = []
    for tail, budget in zip(firing, budgets):
        positions = candidates[tail]
        kept = positions[np.sort(rng.permutation(len(positions))[:budget])]
        tails.append(
            TailSelection(
                tail=tail,
                batch_idx=kept[:, 0],
                token_idx=kept[:, 1],
                n_candidates=len(positions),
                sorted_magnitudes=positive_sorted_activations(
                    columns[tail], 0, scan.valid_mask
                ),
                pick=RandomPick(floor=0.0),
            )
        )
        print(
            f"[neuron {neuron_idx} {tail}] walk keeps {len(positions)} tokens, {len(kept)} kept as hits"
        )
    return merged_selection(config.polarity, tails, scan, neuron_idx)


def main() -> None:
    walk_args, rest = parse_walk_args(sys.argv[1:])
    config = load_run_config(rest)
    stories = tinystories(walk_args.n_stories)
    adapter = GptNeoAdapter.from_pretrained(config.model.value, config.text.max_length)
    picks = saved_or_walked(
        config, adapter.tokenizer, stories, walk_args.share, walk_args.workers
    )

    run = prepare_run(config, adapter, stories)
    with timed("scan"):
        scan = run.capture.scan()
    layer_weight = submodule_getter(config.layer_name)(run.module).weight
    assert isinstance(layer_weight, torch.Tensor)
    weight = run.layout.weight(layer_weight)
    right_padded = adapter.tokenizer.padding_side == "right"
    for column, neuron_idx in enumerate(config.neuron_idxs):
        picked = picked_columns(picks, column, scan.valid_mask, right_padded)
        selection = walk_selection(scan, neuron_idx, picked, config)
        with timed(f"neuron {neuron_idx}: gather"):
            gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
        report_neuron(
            neuron_idx,
            config.polarity,
            selection,
            gathered.inputs,
            scan.sample_ids,
            weight,
            sample_noise(scan, neuron_idx, config.seed),
            run,
            config,
        )


if __name__ == "__main__":
    main()
