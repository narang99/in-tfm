#!/usr/bin/env python
"""Zeroes every neuron the useful-neurons walk did not pick, and compares the logits.

The text is always the start of the attn-only checkpoint's val stream. Other models re-tokenize it.

python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --seq-len 64 --out ablation.html
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --out ablation.html
"""

import argparse
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import plotly.graph_objects as go
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, GPTNeoForCausalLM, PreTrainedTokenizerBase

from in_tfm.models.attn_only_2l.model import load_trained_attn_only
from in_tfm.useful_neurons.attn_only import terms as attn_only_terms
from in_tfm.useful_neurons.attn_only.keep import (
    Keep,
    keep_only,
    with_stream,
    with_stream_and_writers,
)
from in_tfm.useful_neurons.attn_only.trace import trace_sequence
from in_tfm.useful_neurons.corpus import attn_only_tokens, retokenized
from in_tfm.useful_neurons.gpt_neo import keep as gpt_neo_keep
from in_tfm.useful_neurons.gpt_neo import terms as gpt_neo_terms
from in_tfm.useful_neurons.gpt_neo.keep import GptNeoKeep
from in_tfm.useful_neurons.gpt_neo.trace import trace_gpt_neo
from in_tfm.useful_neurons.logit_shift import Logits, LogitShift, logit_shift
from in_tfm.useful_neurons.random_control import kept_shares, random_keep
from in_tfm.useful_neurons.replacement import PersistedValues, Replace, zeros
from in_tfm.useful_neurons.walk import (
    AllLogits,
    find_block_neurons,
    find_useful_neurons,
)

UP = "#2a78d6"
ORIGINAL = "#898781"
DOWN = "#e34948"
NEUTRAL_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
TINYSTORIES_1M = "roneneldan/TinyStories-1M"
Mode = Literal["logits", "blocks", "writers"]


class AblationRun(NamedTuple):
    tokens: torch.Tensor
    tokenizer: PreTrainedTokenizerBase
    before: Logits
    keep: BaseModel
    ablate: Callable[[Any], Logits]
    """Logits with only `keep`'s neurons running. Takes the walk's keep or a random one."""
    n_kept: int


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


def vocab_chart(
    before: Logits,
    after: Logits,
    tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = 20,
) -> go.Figure:
    """One position at a time, picked with a slider: the original's top tokens, before and after."""
    before_probs, after_probs = before.softmax(dim=-1), after.softmax(dim=-1)
    figure = go.Figure(layout=bar_layout("", "probability"))
    titles = []
    for pos in range(len(tokens)):
        top = before_probs[pos].topk(top_k).indices
        names = top_token_labels(top, tokens, pos, tokenizer)
        for name, probs, color in (
            ("original", before_probs, ORIGINAL),
            ("ablated", after_probs, UP),
        ):
            figure.add_bar(
                x=names,
                y=probs[pos, top].tolist(),
                name=name,
                marker_color=color,
                visible=pos == 0,
                hovertemplate=f"%{{x}}<br>{name} %{{y:.4f}}<extra></extra>",
            )
        titles.append(position_title(tokens, pos, tokenizer))

    steps = [
        {
            "label": str(pos),
            "method": "update",
            "args": [
                {"visible": [shown // 2 == pos for shown in range(2 * len(tokens))]},
                {"title": titles[pos]},
            ],
        }
        for pos in range(len(tokens))
    ]
    figure.update_layout(
        title=titles[0],
        barmode="group",
        xaxis_title=f"original's top {top_k} tokens (* = actual next token)",
        sliders=[
            {"steps": steps, "currentvalue": {"prefix": "position "}, "pad": {"t": 130}}
        ],
        height=660,
    )
    return figure


def top_token_labels(
    top: torch.Tensor,
    tokens: torch.Tensor,
    pos: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    actual_next = tokens[pos + 1].item() if pos + 1 < len(tokens) else None
    return [
        f"{tokenizer.decode(token)!r}{' *' if token == actual_next else ''}"
        for token in top.tolist()
    ]


def position_title(
    tokens: torch.Tensor, pos: int, tokenizer: PreTrainedTokenizerBase
) -> str:
    context = tokenizer.decode(tokens[max(0, pos - 8) : pos + 1])
    actual_next = (
        repr(tokenizer.decode(tokens[pos + 1])) if pos + 1 < len(tokens) else "none"
    )
    return f"Position {pos}: ...{context!r}, actual next {actual_next}"


def next_token_labels(
    tokens: torch.Tensor, tokenizer: PreTrainedTokenizerBase
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


def print_kept_shares(keep: BaseModel) -> None:
    shares = kept_shares(keep)
    print(
        "kept per mask: "
        + ", ".join(f"{name} {share:.1%}" for name, share in shares.items())
    )


def print_random_baseline(run: AblationRun, seeds: int) -> None:
    print(f"random baseline, same count per mask, {seeds} seeds:")
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        after = run.ablate(random_keep(run.keep, generator))
        shift = logit_shift(run.before, after, run.tokens)
        same_top = np.mean(
            np.array(shift.top_token_before) == np.array(shift.top_token_after)
        )
        print(
            f"  seed {seed}: mean KL {np.mean(shift.kl):.4f}, top-1 unchanged {same_top:.1%}"
        )


class RunOptions(BaseModel):
    start: int
    seq_len: int
    share: float
    mode: Mode
    recompute_norms: bool
    """TinyStories-1M only. The walk always reads frozen norms; this changes the ablated pass."""
    replace_with: Path | None
    only_mlp: bool
    """TinyStories-1M only. Every attention neuron is kept, so only MLP neurons are dropped."""
    """A `PersistedValues` file for dropped neurons. Without one they become 0."""

    def replace(self) -> Replace:
        if self.replace_with is None:
            return zeros
        return PersistedValues.load(self.replace_with)


def run_attn_only(checkpoint: Path, options: RunOptions) -> AblationRun:
    model, tokenizer = load_trained_attn_only(checkpoint)
    tokens = attn_only_tokens(checkpoint, options.start, options.seq_len)

    original = trace_sequence(model, tokens)
    terms_of = partial(attn_only_terms.terms_of, trace=original)
    share = options.share
    if options.mode == "logits":
        useful = find_useful_neurons(original.unembed, terms_of, AllLogits(), share)
        picked = list(useful.picks)
        keep = keep_only(picked, model.config, len(tokens))
    elif options.mode == "blocks":
        writers = attn_only_terms.block_writers(original)
        picked = list(find_block_neurons(writers, terms_of, share))
        keep = with_stream_and_writers(keep_only(picked, model.config, len(tokens)))
    else:
        writers = attn_only_terms.top_share_writers(original, share)
        picked = list(find_block_neurons(writers, terms_of, share))
        keep = with_stream(keep_only(picked, model.config, len(tokens)))

    replace = options.replace()

    def ablate(keep: Keep) -> Logits:
        return trace_sequence(model, tokens, keep, replace=replace).logits

    return AblationRun(tokens, tokenizer, original.logits, keep, ablate, len(picked))


def run_tinystories(checkpoint: Path, options: RunOptions) -> AblationRun:
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(TINYSTORIES_1M)
    )
    tokens = retokenized(checkpoint, options.start, options.seq_len, tokenizer)
    model = GPTNeoForCausalLM.from_pretrained(TINYSTORIES_1M).eval()

    original = trace_gpt_neo(model, tokens)
    terms_of = partial(gpt_neo_terms.terms_of, trace=original)
    share = options.share
    if options.mode == "logits":
        useful = find_useful_neurons(original.unembed, terms_of, AllLogits(), share)
        picked = list(useful.picks)
        keep = gpt_neo_keep.keep_only(picked, model.config, len(tokens))
    elif options.mode == "blocks":
        writers = gpt_neo_terms.block_writers(original)
        picked = list(find_block_neurons(writers, terms_of, share))
        keep = gpt_neo_keep.with_stream_and_writers(
            gpt_neo_keep.keep_only(picked, model.config, len(tokens))
        )
    else:
        writers = gpt_neo_terms.top_share_writers(original, share)
        picked = list(find_block_neurons(writers, terms_of, share))
        keep = gpt_neo_keep.with_stream(
            gpt_neo_keep.keep_only(picked, model.config, len(tokens))
        )

    if options.only_mlp:
        keep = gpt_neo_keep.with_attention(keep)
    frozen = None if options.recompute_norms else original.frozen_norms
    replace = options.replace()

    def ablate(keep: GptNeoKeep) -> Logits:
        return trace_gpt_neo(model, tokens, keep, frozen=frozen, replace=replace).logits

    return AblationRun(tokens, tokenizer, original.logits, keep, ablate, len(picked))


RUNS = {"attn-only": run_attn_only, "tinystories-1m": run_tinystories}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=Path, help="attn-only checkpoint, also the text source"
    )
    parser.add_argument("--model", choices=list(RUNS), default="attn-only")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--share", type=float, default=0.9)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["logits", "blocks", "writers"],
        default="logits",
        help=(
            "logits: walk back from every logit. "
            "blocks: keep every writer, trim inside blocks. "
            "writers: keep each write vector's top-share dims, trim inside blocks"
        ),
    )
    parser.add_argument(
        "--recompute-norms",
        action="store_true",
        help="tinystories-1m only: the ablated pass recomputes LayerNorm stats",
    )
    parser.add_argument(
        "--only-mlp",
        action="store_true",
        help="tinystories-1m only: keep every attention neuron, drop only MLP neurons",
    )
    parser.add_argument(
        "--replace-with",
        type=Path,
        help="values for dropped neurons, from collect_neuron_means.py. Default 0",
    )
    args = parser.parse_args()
    options = RunOptions(
        start=args.start,
        seq_len=args.seq_len,
        share=args.share,
        mode=args.mode,
        recompute_norms=args.recompute_norms,
        replace_with=args.replace_with,
        only_mlp=args.only_mlp,
    )

    began = time.perf_counter()
    run = RUNS[args.model](args.checkpoint, options)
    print(f"walked and ablated in {time.perf_counter() - began:.0f}s")
    tokens, tokenizer, before = run.tokens, run.tokenizer, run.before
    after = run.ablate(run.keep)
    shift = logit_shift(before, after, tokens)
    print_summary(shift, run.n_kept)
    print_kept_shares(run.keep)
    print_random_baseline(run, args.random_seeds)
    labels = next_token_labels(tokens, tokenizer)
    figures = [
        vocab_chart(before, after, tokens, tokenizer),
        percent_change_chart(shift, labels),
        kl_chart(shift, labels),
    ]
    write_page(figures, args.out)


if __name__ == "__main__":
    main()
