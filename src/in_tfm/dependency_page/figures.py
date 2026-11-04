"""The Plotly figures of a dependency page, in the neuron report's styling."""

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int
from plotly.subplots import make_subplots

from ..cluster_colors import cluster_color
from ..dependencies.stats import SignSplit
from ..report_page.figures import GRID, MARK, MUTED, embed

ADDING = "#00A550"
INHIBITING = "#d33"
ALL = MUTED
HISTOGRAM_BINS = 40
JITTER = 0.3
MAX_STRIP_POINTS = 200
"""Per writer row. A page can hold a hundred cards, and each point is bytes in the page."""
LIGHT_TEMPLATE = "none"
"""Plotly's default template adds about 6 KB to every figure, which adds up over a hundred cards."""


def jitter(n: int) -> Float[np.ndarray, "n"]:
    """Seeded, so a re-render draws the same picture."""
    return np.random.default_rng(0).uniform(-JITTER, JITTER, n).astype(np.float32)


def strip_keep(n: int) -> Int[np.ndarray, "k"]:
    if n <= MAX_STRIP_POINTS:
        return np.arange(n)
    return np.random.default_rng(0).choice(n, MAX_STRIP_POINTS, replace=False)


def point_color(cluster_id: int | None) -> str:
    """Unmatched firings, and writers with no report, stay grey."""
    return MUTED if cluster_id is None else cluster_color(cluster_id)


def histogram_rows(distances: SignSplit, fractions: SignSplit, threshold: float) -> str:
    """Rows are all, adding and inhibiting firings. Left: noise units. Right: per-example fraction past the threshold."""
    rows = [("all", ALL), ("adding", ADDING), ("inhibiting", INHIBITING)]
    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes="columns",
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        column_titles=[
            "writer value, in noise units",
            f"fraction of an example's writers past {threshold}",
        ],
        row_titles=[name for name, _ in rows],
    )
    for row, (name, color) in enumerate(rows, start=1):
        for col, split in ((1, distances), (2, fractions)):
            fig.add_trace(
                go.Histogram(
                    x=np.asarray(getattr(split, name), dtype=np.float32),
                    nbinsx=HISTOGRAM_BINS,
                    marker_color=color,
                    showlegend=False,
                    hovertemplate="%{x}<br>%{y} firings<extra></extra>"
                    if col == 1
                    else "%{x}<br>%{y} examples<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.add_vline(x=threshold, line=dict(color=MARK, dash="dot"), row=row, col=1)
    fig.update_layout(bargap=0.05, template=LIGHT_TEMPLATE)
    return embed(fig, "60vh")


class WriterStrip:
    """One row per shown writer: the observed values as points, coloured by matched cluster, over the noise shorth as a band."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.values: list[list[float]] = []
        self.clusters: list[list[int | None]] = []
        self.bands: list[tuple[float, float]] = []

    def add(
        self,
        label: str,
        units: list[float],
        clusters: list[int | None],
        band: tuple[float, float],
    ) -> None:
        self.labels.append(label)
        self.values.append(units)
        self.clusters.append(clusters)
        self.bands.append(band)

    def html(self) -> str:
        fig = go.Figure()
        for row, (label, units, clusters, (low, high)) in enumerate(
            zip(self.labels, self.values, self.clusters, self.bands)
        ):
            fig.add_shape(
                type="rect",
                x0=low,
                x1=high,
                y0=row - 0.35,
                y1=row + 0.35,
                fillcolor=GRID,
                line_width=0,
                layer="below",
            )
            keep = strip_keep(len(units))
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(units, dtype=np.float32)[keep],
                    y=row + jitter(len(keep)),
                    mode="markers",
                    marker=dict(
                        color=[point_color(clusters[i]) for i in keep],
                        size=4,
                        opacity=0.7,
                    ),
                    text=[
                        "" if clusters[i] is None else f"cluster {clusters[i]}"
                        for i in keep
                    ],
                    hovertemplate="%{x:.2f} noise units %{text}<extra>"
                    + label
                    + "</extra>",
                    showlegend=False,
                )
            )
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        fig.update_yaxes(
            tickvals=list(range(len(self.labels))),
            ticktext=self.labels,
            autorange="reversed",
            automargin=True,
            showgrid=False,
        )
        fig.update_xaxes(title_text="noise units (band: 90% of noise)")
        fig.update_layout(template=LIGHT_TEMPLATE)
        return embed(fig, f"{90 + 34 * len(self.labels)}px")
