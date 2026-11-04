"""The Plotly figures of a dependency page, in the neuron report's styling."""

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots

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


def strip_points(units: list[float]) -> Float[np.ndarray, "k"]:
    values = np.asarray(units, dtype=np.float32)
    if len(values) <= MAX_STRIP_POINTS:
        return values
    keep = np.random.default_rng(0).choice(len(values), MAX_STRIP_POINTS, replace=False)
    return values[keep]


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
    """One row per shown writer: the observed values as points, the noise shorth as a band."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.values: list[list[float]] = []
        self.bands: list[tuple[float, float]] = []

    def add(self, label: str, units: list[float], band: tuple[float, float]) -> None:
        self.labels.append(label)
        self.values.append(units)
        self.bands.append(band)

    def html(self) -> str:
        fig = go.Figure()
        for row, (label, units, (low, high)) in enumerate(
            zip(self.labels, self.values, self.bands)
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
            points = strip_points(units)
            fig.add_trace(
                go.Scatter(
                    x=points,
                    y=row + jitter(len(points)),
                    mode="markers",
                    marker=dict(color=MUTED, size=4, opacity=0.6),
                    hovertemplate="%{x:.2f} noise units<extra>" + label + "</extra>",
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
