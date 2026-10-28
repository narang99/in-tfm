"""Plotly figures as page fragments, styled to sit on the report's light or dark background."""

from collections.abc import Collection

import plotly.graph_objects as go
from markupsafe import escape
from plotly.offline import get_plotlyjs_version

from ..cluster_colors import cluster_color, ink_on

PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
"""The same plotly.js the figures were written for."""

INK = "#8a8a8a"
"""Mid grey reads on both backgrounds, since a figure cannot follow the page's colour scheme."""
GRID = "rgba(138,138,138,0.25)"
MARK = "#e0a800"
"""Cut lines and thresholds."""
MUTED = "#9aa1ae"


def tree_color(tree: int) -> str:
    return cluster_color(tree)


def tree_name(tree: int) -> str:
    return f"c{int(tree)}"


def embed(fig: go.Figure, height: str = "60vh", div_id: str | None = None) -> str:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        margin=dict(l=60, r=20, t=30, b=45),
        autosize=True,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        default_height=height,
        config={"responsive": True, "displaylogo": False},
        div_id=div_id,
    )


def cluster_chip(tree: int, shown: Collection[int], label: str = "", page: str = "") -> str:
    """A chip that links to the cluster's section, or plain when it has none.

    - A cluster gets a section only with enough confident hits, so a tree can be drawn without one.
    - `page` is the index's path from a page beside it, empty on the index itself.
    """
    text = escape(label or tree_name(tree))
    style = f'style="--c:{tree_color(tree)};--ink:{ink_on(tree_color(tree))}"'
    if tree in shown:
        return f'<a class="tree-chip" href="{page}#cluster-{tree}" {style}>{text}</a>'
    return f'<span class="tree-chip unlinked" {style} title="no section: too few confident hits">{text}</span>'
