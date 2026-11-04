#!/usr/bin/env python
"""A 2D cuml UMAP of a neuron report's clustered rows, drawn as one Plotly page.

- The hits and rows are rebuilt with the report's own selection and seed, as in `hdbscan_soft_compare.py`.
- Labels come from that script's `soft.npz`, which holds the hard labels of the same rows in the same order.
- The page draws every clustered hit and a sample of the noise, since a million WebGL points is too heavy to pan.

    python scripts/umap_hadamards.py --config configs/tiny_attn_only_1l_the_logit.yaml
"""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from run_neuron_report import load_adapter, load_samples  # noqa: E402

from in_tfm.cluster_colors import cluster_color  # noqa: E402
from in_tfm.hit_selection import select_hits  # noqa: E402
from in_tfm.neuron_run import clustered_rows, layer_getter_for, prepare_run  # noqa: E402
from in_tfm.run_config import RunConfig, load_run_config  # noqa: E402

N_NOISE_DRAWN = 150_000
UMAP_NEIGHBOURS = 15


class Hits(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    rows: Float[np.ndarray, "n d"]
    activation: Float[np.ndarray, "n"]
    token: list[str]


def hit_tokens(
    texts: list[str],
    adapter,
    batch_idx: Int[np.ndarray, "n"],
    token_idx: Int[np.ndarray, "n"],
    max_length: int,
) -> list[str]:
    """The tokens the report's text source sees: BOS first, cut to `max_length`."""
    tokenizer = adapter.tokenizer
    ids = tokenizer(
        [tokenizer.bos_token + text for text in texts],
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    vocab = {i: tokenizer.decode([i]) for i in range(len(tokenizer))}
    return [vocab[ids[b][t]] for b, t in zip(batch_idx.tolist(), token_idx.tolist())]


def report_hits(config: RunConfig) -> Hits:
    [neuron_idx] = config.neuron_idxs
    adapter = load_adapter(config)
    texts = load_samples(config)
    run = prepare_run(config, adapter, texts)
    scan = run.capture.scan()
    selection = select_hits(scan, neuron_idx, config.polarity, config.hits, config.seed)
    gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)
    return Hits(
        rows=clustered_rows(gathered.inputs, weight, neuron_idx, config.clustering),
        activation=selection.activations,
        token=hit_tokens(
            texts,
            adapter,
            selection.batch_idx,
            selection.token_idx,
            config.text.max_length,
        ),
    )


def umap_2d(rows: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n 2"]:
    from cuml.manifold import UMAP

    return np.asarray(
        UMAP(n_components=2, n_neighbors=UMAP_NEIGHBOURS, random_state=0).fit_transform(
            rows
        )
    )


def drawn_indices(
    labels: Int[np.ndarray, "n"], rng: np.random.Generator
) -> Int[np.ndarray, "m"]:
    noise = np.flatnonzero(labels < 0)
    sampled_noise = rng.choice(noise, min(N_NOISE_DRAWN, len(noise)), replace=False)
    return np.concatenate([sampled_noise, np.flatnonzero(labels >= 0)])


def hover_text(labels, tokens, activation, idx) -> list[str]:
    return [
        f"{'noise' if labels[i] < 0 else f'c{labels[i]}'} · {tokens[i]!r} · logit {activation[i]:+.2f}"
        for i in idx.tolist()
    ]


def figure(embedding, labels, hits: Hits, title: str) -> go.Figure:
    idx = drawn_indices(labels, np.random.default_rng(0))
    noise, clustered = idx[labels[idx] < 0], idx[labels[idx] >= 0]
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=embedding[noise, 0],
            y=embedding[noise, 1],
            mode="markers",
            name=f"noise ({len(noise):,} of {int((labels < 0).sum()):,} drawn)",
            marker=dict(size=2, color="rgba(150,150,150,0.25)"),
            text=hover_text(labels, hits.token, hits.activation, noise),
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=embedding[clustered, 0],
            y=embedding[clustered, 1],
            mode="markers",
            name=f"clustered ({len(clustered):,})",
            marker=dict(
                size=3, color=[cluster_color(int(labels[i])) for i in clustered]
            ),
            text=hover_text(labels, hits.token, hits.activation, clustered),
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        legend=dict(itemsizing="constant"),
        xaxis=dict(title="umap 1"),
        yaxis=dict(title="umap 2", scaleanchor="x"),
    )
    return fig


def main() -> None:
    config = load_run_config(sys.argv[1:])
    started = time.time()
    out = Path(config.out_dir) / "umap"
    labels = np.load(Path(config.out_dir) / "hdbscan_soft" / "soft.npz")["hard"]
    hits = report_hits(config)
    assert len(labels) == len(hits.rows), (len(labels), len(hits.rows))
    print(f"[rows {hits.rows.shape}] {time.time() - started:.1f}s", flush=True)
    embedding = umap_2d(hits.rows)
    print(f"[umap] {time.time() - started:.1f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "embedding.npz",
        embedding=embedding,
        labels=labels,
        activation=hits.activation,
    )
    title = f"UMAP of {config.layer_name} neuron {config.neuron_idxs[0]} rows, {len(labels):,} hits, cuml HDBSCAN labels"
    figure(embedding, labels, hits, title).write_html(
        out / "umap.html", include_plotlyjs="cdn"
    )
    print(f"[done, total] {time.time() - started:.1f}s -> {out / 'umap.html'}")


if __name__ == "__main__":
    main()
