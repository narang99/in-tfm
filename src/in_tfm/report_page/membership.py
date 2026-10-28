"""Soft membership: how split each hit is between clusters, and which clusters touch.

- A hit's cluster is its most probable tree, which can differ from the tree it grew in.
- The same numbers give each cluster card a short stats row.
"""

import math
from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go

from .data import ForestData, touching_counts
from .figures import MARK, cluster_chip, embed, tree_name
from .sections import GuideEntry, PageSection, Pane

NEAREST_SHOWN = 3


def has_threshold(data: ForestData) -> bool:
    return not math.isnan(data.max_entropy)


def moved_hits(data: ForestData) -> int:
    """Hits most probable in a different tree than the one they grew in."""
    return int((data.hit_cluster != data.tree[data.distinct_of_hit]).sum())


def touching_shares(data: ForestData) -> np.ndarray:
    """Row-normalised: cell (a, b) is the share of cluster a's mutual-neighbour pairs that land in b."""
    counts = touching_counts(data).astype(float)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def membership_section(data: ForestData | None) -> PageSection | None:
    if data is None:
        return None
    confident = (
        f" Hits under <b>{data.max_entropy:g}</b> bits are confident, and only they are shown in the clusters below."
        if has_threshold(data) else ""
    )
    return PageSection(
        key="membership",
        title="Soft membership",
        lede_html=(
            "Each hit gets a probability for every cluster, from its graph distance to that cluster's peak. "
            f"<b>{moved_hits(data):,}</b> hits are most probable in a different cluster than the tree they grew in. "
            "From here on, a hit's cluster is its most probable one." + confident
        ),
        panes=[
            Pane(name="Entropy", body_html=embed(entropy_figure(data), "40vh")),
            Pane(
                name="Which clusters touch",
                intro=(
                    "Row is a cluster, column a neighbour. Each cell is the share of the row cluster's mutual "
                    "nearest-neighbour pairs that land in the column cluster. The diagonal is how self-contained a "
                    "cluster is."
                ),
                body_html=embed(touching_figure(data), "70vh"),
            ),
        ],
        guide=[GuideEntry(
            term="Probability and entropy",
            text=(
                "Each hit's probabilities over clusters come from graph distance to each cluster's peak. Entropy in "
                "bits says how split it is: 0 is one cluster, 1 an even split between two."
            ),
        )],
    )


def entropy_figure(data: ForestData) -> go.Figure:
    fig = go.Figure(go.Histogram(x=data.hit_entropy, nbinsx=80, marker_color="#7e57c2"))
    if has_threshold(data):
        fig.add_vline(x=data.max_entropy, line=dict(color=MARK, dash="dash"), annotation_text="confident below")
    fig.update_xaxes(title="entropy (bits)")
    fig.update_yaxes(title="hits")
    return fig


def touching_figure(data: ForestData) -> go.Figure:
    order = np.argsort(-data.hits_per_tree, kind="stable")
    share = touching_shares(data)[np.ix_(order, order)]
    labels = [tree_name(t) for t in order]
    nonzero = share[share > 0]
    top = float(np.percentile(nonzero, 98)) if len(nonzero) else 1.0
    fig = go.Figure(go.Heatmap(
        z=share, x=labels, y=labels, colorscale="Viridis", zmin=0, zmax=top, colorbar=dict(title="share"),
        hovertemplate="from %{y} to %{x}: %{z:.1%} of mutual-neighbour pairs<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="cluster", type="category")
    fig.update_xaxes(title="neighbour cluster", type="category")
    return fig


def cluster_stats_html(data: ForestData | None, cluster_id: int, shown: Collection[int]) -> str:
    """A row of stats for one cluster's card. Empty without a forest, so HDBSCAN cards are unchanged."""
    if data is None or cluster_id >= data.n_trees:
        return ""
    mine = data.hit_cluster == cluster_id
    parts = []
    if has_threshold(data) and mine.any():
        confident = float((data.hit_entropy[mine] < data.max_entropy).mean())
        parts.append(f'<span class="card-stat"><b>{confident:.0%}</b> confident</span>')
    shares = touching_shares(data)[cluster_id]
    parts.append(f'<span class="card-stat"><b>{shares[cluster_id]:.0%}</b> self-contained</span>')
    others = [int(t) for t in np.argsort(-shares) if t != cluster_id and shares[t] > 0][:NEAREST_SHOWN]
    if others:
        chips = "".join(cluster_chip(t, shown, f"{tree_name(t)} {shares[t]:.0%}") for t in others)
        parts.append(f'<span class="card-stat">touches {chips}</span>')
    return f'<div class="card-stats">{"".join(parts)}</div>'
