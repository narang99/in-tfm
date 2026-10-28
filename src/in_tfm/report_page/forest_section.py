"""The "Density-peak forest" section: three views of the forest, as tabs.

- Decision graph: every point by density and distance to anything denser, with both root cuts.
- Root cuts: the sorted rho and delta curves, each with the elbow that set its cut.
- Forest up close: a sampled slice of real points and their paths to the root, laid out on real distances.

Points are distinct points, so a point with copies stands for several hits.
"""

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int
from markupsafe import escape
from plotly.subplots import make_subplots

from .data import ForestData, touching_counts
from .figures import MARK, MUTED, embed, tree_color, tree_name
from .sections import GuideEntry, PageSection, Pane

DEFAULT_PICKS = 2
"""Trees switched on when the up-close tab opens: the pair among the largest trees that touch most."""
UP_CLOSE_DIV = "forest-up-close"


def forest_section(data: ForestData | None) -> PageSection | None:
    if data is None:
        return None
    panes = [
        Pane(name="Decision graph", intro=DECISION_INTRO, body_html=embed(decision_figure(data), "65vh")),
        Pane(name="Root cuts", intro=CUTS_INTRO, body_html=embed(cuts_figure(data), "55vh")),
    ]
    if len(data.up_close_points):
        panes.append(Pane(name="Forest up close", intro=up_close_intro(data), body_html=up_close_html(data)))
    return PageSection(
        key="forest",
        title="Density-peak forest",
        lede_html=(
            "Every point has a density rho, from the distance to its kth neighbour, and a delta, the distance to the "
            f"nearest denser point. Roots pass both cuts, rho &ge; <b>{data.cut_rho:.3f}</b> and "
            f"delta &ge; <b>{data.cut_delta:.3f}</b>: dense, and far from anything denser. "
            f"That gives <b>{data.n_trees}</b> trees."
        ),
        panes=panes,
        guide=[
            GuideEntry(
                term="Point",
                text="Identical hits are merged into one point with a count. Charts in this section draw points.",
            ),
            GuideEntry(
                term="Forest and trees",
                text=(
                    "Each point links to its nearest denser point. Points that are dense and far from anything denser "
                    "become roots, and every other point climbs to one root. Each root's tree is a cluster."
                ),
            ),
        ],
    )


DECISION_INTRO = (
    "Each dot is a point, coloured by the tree it grows into. Right is a crowded neighbourhood, and up means the point "
    "is far from anything denser. Most points sit low, on the slope of some hill. Roots are past both cut lines, in the "
    "top right. High dots on the left are loners, not centres."
)
CUTS_INTRO = (
    "Each cut is the elbow of its own curve, sorted largest first: the point furthest below the line from the first "
    "value to the last. Roots are the points past both cuts."
)


def point_hover(data: ForestData, points: Int[np.ndarray, "n"]) -> list[str]:
    most_probable = data.most_probable
    return [
        f"{tree_name(data.tree[p])} | depth {data.depth[p]} | {data.counts[p]} hit(s) | "
        f"most probable {tree_name(most_probable[p])}"
        for p in points
    ]


def decision_figure(data: ForestData) -> go.Figure:
    points = np.arange(len(data.rho))
    fig = go.Figure(go.Scattergl(
        x=data.rho, y=data.delta, mode="markers", name="points",
        marker=dict(size=4, color=[tree_color(t) for t in data.tree], opacity=0.55),
        text=point_hover(data, points), hovertemplate="%{text}<br>rho %{x:.3f}, delta %{y:.3f}<extra></extra>",
    ))
    trees = np.arange(data.n_trees)
    fig.add_trace(go.Scatter(
        x=data.rho[data.roots], y=data.delta[data.roots], mode="markers+text", name="roots",
        text=[tree_name(t) for t in trees], textposition="top center",
        marker=dict(size=11, color=[tree_color(t) for t in trees], line=dict(color="#fff", width=1.5)),
        hovertemplate="root %{text}<br>rho %{x:.3f}, delta %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(x=data.cut_rho, line=dict(color=MARK, dash="dash"), annotation_text="rho cut")
    fig.add_hline(y=data.cut_delta, line=dict(color=MARK, dash="dash"), annotation_text="delta cut")
    fig.update_xaxes(title="rho (log density)")
    fig.update_yaxes(title="delta (distance to the nearest denser point)")
    fig.update_layout(showlegend=False)
    return fig


def delta_for_elbow(data: ForestData) -> Float[np.ndarray, "n_distinct"]:
    """As `density_peaks.delta_for_elbow`: the densest point's delta is capped at the next largest."""
    delta = data.delta.copy()
    densest = int(np.argmax(data.rho))
    delta[densest] = np.delete(delta, densest).max(initial=delta[densest])
    return delta


def cuts_figure(data: ForestData) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("rho, sorted", "delta, sorted"), horizontal_spacing=0.08)
    for col, (values, cut) in enumerate([(data.rho, data.cut_rho), (delta_for_elbow(data), data.cut_delta)], start=1):
        curve = np.sort(values)[::-1]
        fig.add_trace(go.Scattergl(
            x=np.arange(len(curve)), y=curve, mode="lines", line=dict(color=MUTED, width=2),
            hovertemplate="rank %{x}<br>%{y:.3f}<extra></extra>",
        ), row=1, col=col)
        fig.add_hline(y=cut, line=dict(color=MARK, dash="dash"), annotation_text=f"cut {cut:.3f}", row=1, col=col)
        fig.add_vline(x=int((curve >= cut).sum()) - 1, line=dict(color=MARK, dash="dot", width=1), row=1, col=col)
    fig.update_xaxes(title="points, largest first")
    fig.update_layout(showlegend=False)
    return fig


def default_picks(data: ForestData) -> set[int]:
    touching = touching_counts(data).astype(float)
    np.fill_diagonal(touching, 0)
    largest = np.argsort(-data.hits_per_tree, kind="stable")[:8]
    sub = touching[np.ix_(largest, largest)]
    if data.n_trees < DEFAULT_PICKS or sub.max() == 0:
        return set(map(int, largest[:DEFAULT_PICKS]))
    return {int(largest[i]) for i in np.unravel_index(np.argmax(sub), sub.shape)}


def up_close_intro(data: ForestData) -> str:
    return (
        f"A slice of the real forest: up to {data.up_close_per_tree} random points from every tree, plus every point on "
        "their path to the root, laid out once on real distances. Near means similar, and positions stay put as trees "
        "are added. Lines join a point to its parent, bigger dots are denser, and stars are roots. It opens on the two "
        "large trees that touch most."
    )


def up_close_html(data: ForestData) -> str:
    picks = default_picks(data)
    buttons = "".join(
        f'<button class="pick{" on" if t in picks else ""}" data-t="{t}" style="--c:{tree_color(t)}" '
        f'onclick="togglePick(this)">{escape(tree_name(t))}</button>'
        for t in np.argsort(-data.hits_per_tree, kind="stable")
    )
    picker = (
        f'<div class="picker" data-plot="{UP_CLOSE_DIV}" data-trees="{data.n_trees}">'
        '<button class="ctl" onclick="pickAll(this, true)">all</button>'
        '<button class="ctl" onclick="pickAll(this, false)">none</button>'
        f"{buttons}</div>"
    )
    return picker + embed(up_close_figure(data, picks), "75vh", div_id=UP_CLOSE_DIV)


def up_close_figure(data: ForestData, picks: set[int]) -> go.Figure:
    """Three traces per tree, in tree order: links, points, root. `page_tabs.js` relies on that order."""
    points, xy = data.up_close_points, data.up_close_xy
    slot = {int(p): i for i, p in enumerate(points)}
    rho = data.rho[points]
    size = 4 + 14 * (rho - rho.min()) / max(float(np.ptp(rho)), 1e-9)
    fig = go.Figure()
    for t in range(data.n_trees):
        on = t in picks
        mine = np.array([i for i, p in enumerate(points) if data.tree[p] == t], dtype=int)
        linked = np.array([i for i in mine if data.parent_in_tree[points[i]] >= 0], dtype=int)
        parents = np.array([slot[int(data.parent_in_tree[points[i]])] for i in linked], dtype=int)
        gap = np.full(len(linked), np.nan)
        fig.add_trace(go.Scatter(
            x=np.column_stack([xy[linked, 0], xy[parents, 0], gap]).ravel() if len(linked) else [],
            y=np.column_stack([xy[linked, 1], xy[parents, 1], gap]).ravel() if len(linked) else [],
            mode="lines", line=dict(color=tree_color(t), width=1), opacity=0.6, hoverinfo="skip", visible=on,
        ))
        fig.add_trace(go.Scatter(
            x=xy[mine, 0], y=xy[mine, 1], mode="markers", visible=on,
            marker=dict(size=size[mine], color=tree_color(t), line=dict(color="#000", width=0.5)),
            text=point_hover(data, points[mine]), hovertemplate="%{text}<extra></extra>",
        ))
        root = slot.get(int(data.roots[t]))
        fig.add_trace(go.Scatter(
            x=[xy[root, 0]] if root is not None else [], y=[xy[root, 1]] if root is not None else [],
            mode="markers+text", text=[tree_name(t)], textposition="top center", visible=on, hoverinfo="skip",
            marker=dict(symbol="star", size=20, color=tree_color(t), line=dict(color="#fff", width=1.5)),
        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(showlegend=False)
    return fig
