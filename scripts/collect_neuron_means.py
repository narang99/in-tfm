#!/usr/bin/env python
"""Each neuron's mean activation over the val stream, saved for `ablate_unpicked_neurons.py --replace-with`.

python scripts/collect_neuron_means.py checkpoints/tiny_attn_only_1l_d128 --out means_1l.pt
python scripts/collect_neuron_means.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --out means_ts1m.pt
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoForCausalLM, PreTrainedTokenizerBase

from in_tfm.models.attn_only_2l.model import load_trained_attn_only
from in_tfm.useful_neurons.attn_only import trace as attn_only_trace
from in_tfm.useful_neurons.corpus import attn_only_tokens, retokenized
from in_tfm.useful_neurons.gpt_neo import trace as gpt_neo_trace
from in_tfm.useful_neurons.neuron_means import MeanAccumulator, SiteActivations

TINYSTORIES_1M = "roneneldan/TinyStories-1M"

WindowActivations = Callable[[torch.Tensor], SiteActivations]


def attn_only_stream(
    checkpoint: Path, n_tokens: int
) -> tuple[torch.Tensor, WindowActivations, int]:
    model, _ = load_trained_attn_only(checkpoint)

    def activations(window: torch.Tensor) -> SiteActivations:
        trace = attn_only_trace.trace_sequence(model, window)
        return attn_only_trace.site_activations(trace)

    tokens = attn_only_tokens(checkpoint, 0, n_tokens)
    return tokens, activations, model.config.num_hidden_layers


def tinystories_stream(
    checkpoint: Path, n_tokens: int
) -> tuple[torch.Tensor, WindowActivations, int]:
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(TINYSTORIES_1M)
    )
    model = GPTNeoForCausalLM.from_pretrained(TINYSTORIES_1M).eval()

    def activations(window: torch.Tensor) -> SiteActivations:
        return gpt_neo_trace.site_activations(
            gpt_neo_trace.trace_gpt_neo(model, window)
        )

    tokens = retokenized(checkpoint, 0, n_tokens, tokenizer)
    return tokens, activations, model.config.num_layers


STREAMS = {"attn-only": attn_only_stream, "tinystories-1m": tinystories_stream}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=Path, help="attn-only checkpoint, also the text source"
    )
    parser.add_argument("--model", choices=list(STREAMS), default="attn-only")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokens, activations, n_layers = STREAMS[args.model](args.checkpoint, args.tokens)
    accumulator = MeanAccumulator()
    windows = tokens[: len(tokens) // args.window * args.window].view(-1, args.window)
    for window in tqdm(windows, desc="windows"):
        accumulator.add(activations(window), n_tokens=args.window)
    accumulator.means(n_layers).save(args.out)
    print(f"means over {accumulator.tokens} tokens saved to {args.out}")


if __name__ == "__main__":
    main()
