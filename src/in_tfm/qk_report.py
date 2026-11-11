"""The report page of one head's clustered query-key pairs."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Int
from pydantic import BaseModel

from .cluster_membership import NOISE_LABEL
from .html_report import page, render_template
from .qk_pairs import PairRows

SpanKind = Literal["key", "query", "both", "plain", "gap"]


class Span(BaseModel):
    kind: SpanKind
    text: str = ""


class PairExample(BaseModel):
    sample_id: str
    key_idx: int
    query_idx: int
    score: float
    spans: list[Span]


class TokenCount(BaseModel):
    text: str
    count: int


class DimValue(BaseModel):
    dim: int
    value: float


class PairCluster(BaseModel):
    cluster_id: int
    n_hits: int
    mean_score: float
    median_distance: int
    first_key_share: float
    self_share: float
    top_keys: list[TokenCount]
    top_queries: list[TokenCount]
    top_dims: list[DimValue]
    examples: list[PairExample]


def window(lo: int, hi: int, first_real: int) -> range:
    return range(max(lo, first_real), hi)


def pair_spans(
    tokens: Sequence[str], key: int, query: int, context: int, first_real: int
) -> list[Span]:
    """Context before the key, then the query with context on both sides; a gap marks what is skipped."""
    shown = sorted(
        set(window(key - context, key + 1, first_real))
        | set(window(query - context, min(query + 4, len(tokens)), first_real))
    )
    spans: list[Span] = []
    for previous, position in zip([None, *shown], shown):
        if previous is not None and position != previous + 1:
            spans.append(Span(kind="gap"))
        spans.append(Span(kind=span_kind(position, key, query), text=tokens[position]))
    return spans


def span_kind(position: int, key: int, query: int) -> SpanKind:
    if position == key == query:
        return "both"
    if position == key:
        return "key"
    if position == query:
        return "query"
    return "plain"


def top_tokens(ids: Sequence[int], decode, n: int = 10) -> list[TokenCount]:
    return [
        TokenCount(text=decode([token]), count=count)
        for token, count in Counter(ids).most_common(n)
    ]


def top_dims(mean_row: torch.Tensor, n: int = 8) -> list[DimValue]:
    order = mean_row.abs().argsort(descending=True)[:n]
    return [DimValue(dim=int(d), value=float(mean_row[d])) for d in order]


def cluster_order(labels: Int[np.ndarray, "n_hits"]) -> list[int]:
    """Largest first, with noise last."""
    counts = Counter(labels.tolist())
    return sorted(counts, key=lambda c: (c == NOISE_LABEL, -counts[c]))


class PairReportInputs(BaseModel):
    """Everything the page reads, saved next to it so the page can be rebuilt without the model."""

    model_config = {"arbitrary_types_allowed": True}

    pair_rows: PairRows
    labels: Int[np.ndarray, "n_hits"]
    sample_ids: list[str]
    token_ids: Int[torch.Tensor, "batch seq"]


def first_real_positions(
    token_ids: torch.Tensor, pad_id: int
) -> Int[np.ndarray, "batch"]:
    """Left padding puts real tokens after the pads."""
    real = (token_ids != pad_id).int()
    return real.argmax(dim=1).numpy()


def build_clusters(
    inputs: PairReportInputs,
    decode,
    pad_id: int,
    examples_per_cluster: int,
    context: int,
    seed: int,
) -> list[PairCluster]:
    hits = inputs.pair_rows.hits
    scores = inputs.pair_rows.scores
    sample_idx = torch.from_numpy(hits.sample_idx)
    key_ids = inputs.token_ids[sample_idx, torch.from_numpy(hits.key_idx)].tolist()
    query_ids = inputs.token_ids[sample_idx, torch.from_numpy(hits.query_idx)].tolist()
    first_real = first_real_positions(inputs.token_ids, pad_id)
    rng = np.random.default_rng(seed)
    clusters = []
    for cluster_id in cluster_order(inputs.labels):
        members = np.flatnonzero(inputs.labels == cluster_id)
        distance = hits.query_idx[members] - hits.key_idx[members]
        shown = rng.choice(
            members, min(examples_per_cluster, len(members)), replace=False
        )
        clusters.append(
            PairCluster(
                cluster_id=cluster_id,
                n_hits=len(members),
                mean_score=float(scores[members].mean()),
                median_distance=int(np.median(distance)),
                first_key_share=float(
                    (
                        hits.key_idx[members] == first_real[hits.sample_idx[members]]
                    ).mean()
                ),
                self_share=float((distance == 0).mean()),
                top_keys=top_tokens([key_ids[i] for i in members], decode),
                top_queries=top_tokens([query_ids[i] for i in members], decode),
                top_dims=top_dims(inputs.pair_rows.rows[members].mean(0)),
                examples=[
                    example(inputs, i, decode, context, first_real) for i in shown
                ],
            )
        )
    return clusters


def example(
    inputs: PairReportInputs,
    hit: int,
    decode,
    context: int,
    first_real: Int[np.ndarray, "batch"],
) -> PairExample:
    hits = inputs.pair_rows.hits
    sample = int(hits.sample_idx[hit])
    tokens = [decode([t]) for t in inputs.token_ids[sample].tolist()]
    key, query = int(hits.key_idx[hit]), int(hits.query_idx[hit])
    return PairExample(
        sample_id=inputs.sample_ids[sample],
        key_idx=key,
        query_idx=query,
        score=float(inputs.pair_rows.scores[hit]),
        spans=pair_spans(tokens, key, query, context, int(first_real[sample])),
    )


def write_pair_report(
    out_dir: Path,
    title: str,
    inputs: PairReportInputs,
    decode,
    pad_id: int,
    examples_per_cluster: int,
    context: int,
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, out_dir / "pairs.pt")
    hits = inputs.pair_rows.hits
    body = render_template(
        "qk_pairs.html",
        title=title,
        n_candidates=hits.n_candidates,
        cutoff=hits.cutoff,
        n_hits=len(hits.sample_idx),
        clusters=build_clusters(
            inputs, decode, pad_id, examples_per_cluster, context, seed
        ),
    )
    path = out_dir / "index.html"
    path.write_text(page(title, body))
    return path
