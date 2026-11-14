#!/usr/bin/env python
"""Walks back from every logit of one sequence and prints how many neurons of each kind were picked.

python scripts/find_useful_neurons.py checkpoints/tiny_attn_only_1l_d128 --seq-len 32
"""

import argparse
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from in_tfm.models.attn_only_2l.model import load_trained_attn_only
from in_tfm.useful_neurons.attention import all_scores
from in_tfm.useful_neurons.attn_only.terms import terms_of
from in_tfm.useful_neurons.attn_only.trace import Trace, trace_sequence
from in_tfm.useful_neurons.walk import AllLogits, UsefulNeurons, find_useful_neurons


def val_sequence(checkpoint: Path, start: int, seq_len: int) -> torch.Tensor:
    tokens = np.load(checkpoint / "val_tokens.npy", mmap_mode="r")
    return torch.from_numpy(tokens[start : start + seq_len].astype(np.int64))


def neuron_counts(trace: Trace) -> dict[str, int]:
    seq, hidden = trace.resid[0].shape
    layer = trace.layers[0]
    heads, _, head_dim = layer.q.shape
    n_layers = len(trace.layers)
    per_head = n_layers * seq * heads * head_dim
    return {
        "resid": (n_layers + 1) * seq * hidden,
        "pos": seq * hidden,
        "o_proj": n_layers * seq * heads * hidden,
        "head_out": per_head,
        "v": per_head,
        "q": per_head,
        "k": per_head,
        "score": n_layers * heads * seq * (seq + 1) // 2,
    }


def print_summary(useful: UsefulNeurons, trace: Trace) -> None:
    picked: Counter[str] = Counter(neuron.kind for neuron in useful.picks)
    print(f"{'kind':<10}{'picked':>10}{'of':>10}{'share':>8}")
    for kind, total in neuron_counts(trace).items():
        print(f"{kind:<10}{picked[kind]:>10}{total:>10}{picked[kind] / total:>8.1%}")
    logit_dims = useful.logit_picks.sum(dim=-1).float()
    print(f"final resid dims kept per logit: mean {logit_dims.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--share", type=float, default=0.9)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model, tokenizer = load_trained_attn_only(args.checkpoint)
    tokens = val_sequence(args.checkpoint, args.start, args.seq_len)
    print(repr(tokenizer.decode(tokens)))

    began = time.perf_counter()
    trace = trace_sequence(model, tokens)
    neuron_terms = partial(terms_of, trace=trace)
    scores = all_scores(trace.layers)
    useful = find_useful_neurons(
        trace.unembed, neuron_terms, AllLogits(), args.share, scores
    )
    print(f"walked in {time.perf_counter() - began:.1f}s")
    print_summary(useful, trace)
    if args.out:
        torch.save(useful, args.out)


if __name__ == "__main__":
    main()
