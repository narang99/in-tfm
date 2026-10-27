"""The "Density-peak forest" section: four views of the forest, as tabs.

- Decision graph: every point by density and distance to anything denser, with both root cuts.
- Peak separation: how far each root stands above the delta cut, next to the best points that missed it.
- Forest up close: a sampled slice of real points and their paths to the root, laid out on real distances.
- Forest landscape: each tree as a hill, density against the order of points along the tree.

Points are distinct points, so a point with copies stands for several hits.
"""

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Int
from markupsafe import escape

from .data import ForestData, touching_counts
from .figures import MARK, MUTED, embed, tree_color, tree_name
from .sections import GuideEntry, PageSection, Pane

RUNNERS_UP = 10
DEFAULT_PICKS = 2
"""Trees switched on when the up-close tab opens: the pair among the largest trees that touch most."""
UP_CLOSE_DIV = "forest-up-close"


def forest_section(data: ForestData | None) -> PageSection | None:
    if data is None:
        return None
    panes = [
        Pane(name="Decision graph", intro=DECISION_INTRO, body_html=embed(decision_figure(data), "65vh")),
        Pane(name="Peak separation", intro=SEPARATION_INTRO, body_html=embed(separation_figure(data), "55vh")),
    ]
    if len(data.up_close_points):
        panes.append(Pane(name="Forest up close", intro=up_close_intro(data), body_html=up_close_html(data)))
    panes.append(Pane(name="Forest landscape", intro=LANDSCAPE_INTRO, body_html=embed(landscape_figure(data), "70vh")))
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
SEPARATION_INTRO = (
    "How clearly each peak stands out. Every root is dense enough, and its bar is how far it is from anything denser. "
    "Tall bars are clear peaks. Bars just above the cut would merge under a slightly stricter cut. Grey bars are the "
    "densest-enough points that missed it."
)
LANDSCAPE_INTRO = (
    "Each tree drawn as a hill: height is density, lines are parent links, and the root is on top. Across is only the "
    "order of points along their tree, not a distance. Click a tree in the legend to isolate it, and double-click to "
    "add others."
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


def runners_up(data: ForestData) -> Int[np.ndarray, "n_runners"]:
    is_root = np.zeros(len(data.rho), bool)
    is_root[data.roots] = True
    candidates = np.flatnonzero(~is_root & data.is_root_candidate.astype(bool))
    return candidates[np.argsort(-data.delta[candidates])][:RUNNERS_UP]


def separation_figure(data: ForestData) -> go.Figure:
    """The axis starts at 0: a runner-up is below the delta cut by construction, so a raised floor would hide it."""
    tree_of_root = {int(r): t for t, r in enumerate(data.roots)}
    shown = sorted([*map(int, data.roots), *map(int, runners_up(data))], key=lambda p: -data.delta[p])
    fig = go.Figure(go.Bar(
        x=[tree_name(tree_of_root[p]) if p in tree_of_root else f"#{p}" for p in shown],
        y=[data.delta[p] for p in shown],
        marker_color=[tree_color(tree_of_root[p]) if p in tree_of_root else MUTED for p in shown],
        text=[f"rho {data.rho[p]:.3f} | {data.counts[p]} hit(s)" for p in shown], textposition="none",
        hovertemplate="%{x}<br>delta %{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.add_hline(y=data.cut_delta, line=dict(color=MARK, dash="dash"), annotation_text="delta cut")
    fig.update_yaxes(title="delta (distance to the nearest denser point)", rangemode="tozero")
    fig.update_xaxes(title="roots, coloured, and the best points that missed the cut, grey", type="category")
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


def tree_order(data: ForestData) -> Int[np.ndarray, "n_distinct"]:
    """Depth first from each root, sparsest child first, so each tree draws as one hill."""
    children: dict[int, list[int]] = {}
    for point, parent in enumerate(data.parent_in_tree):
        if parent >= 0:
            children.setdefault(int(parent), []).append(point)
    order: list[int] = []
    for root in data.roots:
        stack = [int(root)]
        while stack:
            point = stack.pop()
            order.append(point)
            stack.extend(sorted(children.get(point, []), key=lambda child: data.rho[child]))
    return np.array(order, dtype=int)


def landscape_figure(data: ForestData) -> go.Figure:
    """Each tree's root marker shares its legend group, so isolating a tree keeps its root."""
    x = np.empty(len(data.rho))
    x[tree_order(data)] = np.arange(len(data.rho))
    hits = data.hits_per_tree
    fig = go.Figure()
    for t in range(data.n_trees):
        members = np.flatnonzero(data.tree == t)
        linked = members[data.parent_in_tree[members] >= 0]
        parents = data.parent_in_tree[linked]
        gap = np.full(len(linked), np.nan)
        group = tree_name(t)
        fig.add_trace(go.Scattergl(
            x=np.column_stack([x[linked], x[parents], gap]).ravel(),
            y=np.column_stack([data.rho[linked], data.rho[parents], gap]).ravel(),
            mode="lines", line=dict(color=tree_color(t), width=0.7), opacity=0.5, legendgroup=group,
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=x[members], y=data.rho[members], mode="markers", name=f"{group} ({hits[t]:,})", legendgroup=group,
            marker=dict(size=3, color=tree_color(t)), text=point_hover(data, members),
            hovertemplate="%{text}<extra></extra>",
        ))
        root = data.roots[t]
        fig.add_trace(go.Scatter(
            x=[x[root]], y=[data.rho[root]], mode="markers+text", text=[group], textposition="top center",
            legendgroup=group, showlegend=False, hoverinfo="skip",
            marker=dict(size=10, symbol="triangle-up", color=tree_color(t), line=dict(color="#fff", width=1)),
        ))
    fig.update_xaxes(title="points, in order along their tree", showticklabels=False)
    fig.update_yaxes(title="rho (log density)")
    fig.update_layout(legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=11)))
    return fig
