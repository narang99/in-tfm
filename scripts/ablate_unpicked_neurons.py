#!/usr/bin/env python
"""Zeroes every neuron the useful-neurons walk did not pick, and compares the logits.

python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --seq-len 64 --out ablation.html
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from transformers import PreTrainedTokenizerFast

from in_tfm.models.attn_only_2l.model import load_trained_attn_only
from in_tfm.useful_neurons.keep import keep_only
from in_tfm.useful_neurons.logit_shift import LogitShift, logit_shift, trace_logits
from in_tfm.useful_neurons.trace import trace_sequence
from in_tfm.useful_neurons.walk import AllLogits, find_useful_neurons

UP = "#2a78d6"
DOWN = "#e34948"
NEUTRAL_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def val_sequence(checkpoint: Path, start: int, seq_len: int) -> torch.Tensor:
    tokens = np.load(checkpoint / "val_tokens.npy", mmap_mode="r")
    return torch.from_numpy(tokens[start : start + seq_len].astype(np.int64))


def bar_layout(title: str, y_title: str) -> go.Layout:
    return go.Layout(
        title=title,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font={"color": NEUTRAL_INK},
        xaxis={"title": "position (next token)", "tickangle": -60},
        yaxis={"title": y_title, "gridcolor": GRID, "zerolinecolor": NEUTRAL_INK},
        bargap=0.15,
        height=420,
    )


def percent_change_chart(shift: LogitShift, labels: list[str]) -> go.Figure:
    change = shift.percent_change
    hover = [
        f"{label}<br>before {before:.4f}<br>after {after:.4f}<br>{pct:+.1f}%"
        for label, before, after, pct in zip(
            labels, shift.next_token_prob_before, shift.next_token_prob_after, change
        )
    ]
    bars = go.Bar(
        x=labels,
        y=change,
        marker_color=[UP if pct >= 0 else DOWN for pct in change],
        hovertext=hover,
        hoverinfo="text",
    )
    title = "Change in the next token's probability after zeroing unpicked neurons"
    return go.Figure(bars, bar_layout(title, "% change from original"))


def kl_chart(shift: LogitShift, labels: list[str]) -> go.Figure:
    bars = go.Bar(
        x=labels,
        y=shift.kl[:-1],
        marker_color=UP,
        hovertemplate="%{x}<br>KL %{y:.4f} nats<extra></extra>",
    )
    title = "KL(original || ablated) per position"
    return go.Figure(bars, bar_layout(title, "KL (nats)"))


def next_token_labels(
    tokens: torch.Tensor, tokenizer: PreTrainedTokenizerFast
) -> list[str]:
    """Plotly needs unique category labels, so each carries its position."""
    return [
        f"{pos}: {tokenizer.decode(token)!r}" for pos, token in enumerate(tokens[1:])
    ]


def write_page(figures: list[go.Figure], out: Path) -> None:
    parts = [figures[0].to_html(full_html=False, include_plotlyjs="cdn")]
    parts += [
        fig.to_html(full_html=False, include_plotlyjs=False) for fig in figures[1:]
    ]
    body = "\n".join(parts)
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Unpicked ablation</title></head>"
        f"<body style='background:{SURFACE}'>{body}</body></html>"
    )


def print_summary(shift: LogitShift, n_kept: int) -> None:
    same_top = np.mean(
        np.array(shift.top_token_before) == np.array(shift.top_token_after)
    )
    print(f"neurons kept: {n_kept}")
    print(f"mean KL: {np.mean(shift.kl):.4f} nats, max {np.max(shift.kl):.4f}")
    print(f"top-1 token unchanged at {same_top:.1%} of positions")
    print(f"median next-token change: {np.median(shift.percent_change):+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--share", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load_trained_attn_only(args.checkpoint)
    tokens = val_sequence(args.checkpoint, args.start, args.seq_len)

    original = trace_sequence(model, tokens)
    useful = find_useful_neurons(original, AllLogits(), args.share)
    picked = list(useful.picks)
    ablated = trace_sequence(
        model, tokens, keep_only(picked, model.config, len(tokens))
    )

    shift = logit_shift(
        trace_logits(original, model), trace_logits(ablated, model), tokens
    )
    print_summary(shift, len(picked))
    labels = next_token_labels(tokens, tokenizer)
    write_page([percent_change_chart(shift, labels), kl_chart(shift, labels)], args.out)


if __name__ == "__main__":
    main()
