"""The "Map" section: a sample of hits, each pulled toward the peaks of the clusters it belongs to."""

import numpy as np
import plotly.graph_objects as go

from .data import ForestData
from .figures import embed, tree_color, tree_name
from .sections import GuideEntry, PageSection, Pane

MAP_DIV = "hit-map"
TOP_PROBABILITIES = 3


def map_section(data: ForestData | None) -> PageSection | None:
    if data is None or not len(data.map_points):
        return None
    return PageSection(
        key="map",
        title="Map",
        lede_html=(
            f"A sample of {len(data.map_points):,} points, laid out on real distances, with each one pulled toward the "
            "peak of every cluster it has some probability in. A confident point sits near its peak, and a point split "
            "between two clusters sits between them. Stars are peaks. Hover a point for its probabilities, click it to "
            "jump to its cluster, and click a cluster in the legend to isolate it."
        ),
        panes=[Pane(name="By cluster", body_html=embed(map_figure(data), "75vh", div_id=MAP_DIV))],
        guide=[GuideEntry(
            term="Map",
            text="Positions are a spring layout, so only nearness means anything. The axes have no units.",
        )],
    )


def probability_text(row: np.ndarray) -> str:
    top = np.argsort(-row)[:TOP_PROBABILITIES]
    return ", ".join(f"{tree_name(t)} {row[t]:.0%}" for t in top if row[t] > 0)


def map_figure(data: ForestData) -> go.Figure:
    """One trace per cluster, so the legend can isolate it. `customdata` carries the cluster for the click handler."""
    points, xy = data.map_points, data.map_xy
    cluster = data.most_probable[points]
    size = 4 + 2 * np.log1p(data.counts[points])
    fig = go.Figure()
    for t in np.argsort(-data.hits_per_tree, kind="stable"):
        mine = np.flatnonzero(cluster == t)
        if not len(mine):
            continue
        fig.add_trace(go.Scattergl(
            x=xy[mine, 0], y=xy[mine, 1], mode="markers", name=f"{tree_name(t)} ({data.hits_per_tree[t]:,})",
            legendgroup=tree_name(t), marker=dict(size=size[mine], color=tree_color(t), opacity=0.75),
            customdata=np.full(len(mine), int(t)),
            text=[f"{data.counts[points[i]]} hit(s) | {probability_text(data.probabilities[points[i]])}" for i in mine],
            hovertemplate="%{text}<extra></extra>",
        ))
    slot = {int(p): i for i, p in enumerate(points)}
    for t, root in enumerate(data.roots):
        i = slot[int(root)]
        fig.add_trace(go.Scatter(
            x=[xy[i, 0]], y=[xy[i, 1]], mode="markers+text", text=[tree_name(t)], textposition="top center",
            legendgroup=tree_name(t), showlegend=False, customdata=[t], hovertemplate=f"peak of {tree_name(t)}<extra></extra>",
            marker=dict(symbol="star", size=16, color=tree_color(t), line=dict(color="#fff", width=1)),
        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=11)))
    return fig
