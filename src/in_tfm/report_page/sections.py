"""The numbered sections of a neuron's report page.

- A section is a title, a lede, and one or more panes shown as tabs.
- Each section also contributes terms to the page's glossary.
- A builder returns None when its data is missing, so the page only numbers what it shows.
"""

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel

from ..activation_ranges import ActivationRanges
from .data import SelectionCurve
from .figures import MARK, MUTED, embed, tree_color, tree_name


class GuideEntry(BaseModel):
    term: str
    text: str


class Pane(BaseModel):
    name: str
    intro: str = ""
    body_html: str


class PageSection(BaseModel):
    key: str
    """Unique per page, used for the tab group and the anchor."""
    title: str
    lede_html: str
    panes: list[Pane]
    guide: list[GuideEntry] = []


class StatTile(BaseModel):
    value: str
    name: str


def activations_section(curve: SelectionCurve | None, ranges: ActivationRanges | None) -> PageSection | None:
    if curve is None:
        return None
    panes = [Pane(name="Sorted curve", body_html=embed(sorted_curve_figure(curve), "55vh"))]
    if ranges is not None and ranges.noise:
        panes.append(Pane(
            name="Histogram",
            intro=(
                f"A random sample of {len(ranges.noise):,} positions, hits or not, signed. "
                "Counts are on a log scale. The shaded tail is past the threshold."
            ),
            body_html=embed(noise_histogram_figure(np.asarray(ranges.noise), curve.threshold, curve.negative), "50vh"),
        ))
    return PageSection(
        key="activations",
        title="Activations and the threshold",
        lede_html=(
            f"{curve.n_positive:,} positions fire in this tail. Sorted low to high, most sit in a long flat stretch "
            f"and a few shoot up at the end. Hits are the positions at or past <b>{curve.threshold:.4g}</b>."
        ),
        panes=panes,
        guide=[GuideEntry(
            term="Hit",
            text="A position where the neuron fires past the threshold. Only hits are clustered.",
        )],
    )


def sorted_curve_figure(curve: SelectionCurve) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=curve.index, y=curve.magnitude, mode="lines", name="sorted activations", line=dict(width=2.5),
    ))
    if curve.elbow_idx is not None:
        add_elbow(fig, curve)
    if curve.bucket_edges is not None:
        add_bucket_edges(fig, curve)
    fig.update_xaxes(title="sorted index (low to high)")
    fig.update_yaxes(title="activation magnitude")
    fig.update_layout(legend=dict(x=0.01, y=0.99))
    return fig


def add_elbow(fig: go.Figure, curve: SelectionCurve) -> None:
    """The elbow is the point furthest below the chord from the first value to the last."""
    first, last, n = float(curve.magnitude[0]), float(curve.magnitude[-1]), curve.n_positive
    elbow = curve.elbow_idx
    chord_at_elbow = first + (last - first) * elbow / max(n - 1, 1)
    fig.add_vrect(
        x0=elbow, x1=n - 1, fillcolor=MARK, opacity=0.1, line_width=0,
        annotation_text="hits", annotation_position="top left",
    )
    fig.add_trace(go.Scatter(
        x=[0, n - 1], y=[first, last], mode="lines", name="chord, first to last", line=dict(color=MUTED, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=[elbow, elbow], y=[curve.threshold, chord_at_elbow], mode="lines", name="largest gap to the chord",
        line=dict(color="#ff7f50", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[elbow], y=[curve.threshold], mode="markers", name=f"elbow: {curve.threshold:.4g}",
        marker=dict(color=MARK, size=12),
    ))


def add_bucket_edges(fig: go.Figure, curve: SelectionCurve) -> None:
    """Edges are activation values, so each sits at the first shown value at or above it. The last edge is the max."""
    for edge in curve.bucket_edges[:-1]:
        position = int(curve.index[min(np.searchsorted(curve.magnitude, edge), len(curve.index) - 1)])
        fig.add_vline(x=position, line=dict(color=MARK, dash="dash", width=1), annotation_text=f"{edge:.3g}")


def noise_histogram_figure(noise: np.ndarray, threshold: float, negative: bool) -> go.Figure:
    fig = go.Figure(go.Histogram(x=noise, nbinsx=120, marker_color=MUTED))
    cut = -threshold if negative else threshold
    tail_end = min(float(noise.min()), cut) if negative else max(float(noise.max()), cut)
    fig.add_vrect(x0=min(cut, tail_end), x1=max(cut, tail_end), fillcolor=MARK, opacity=0.12, line_width=0)
    fig.add_vline(x=cut, line=dict(color=MARK, dash="dash"), annotation_text="threshold")
    fig.update_yaxes(type="log", title="positions (log scale)")
    fig.update_xaxes(title="activation")
    fig.update_layout(showlegend=False)
    return fig


def cluster_ranges_section(ranges: ActivationRanges | None, cluster_ids: list[int]) -> PageSection | None:
    if ranges is None or not ranges.clusters:
        return None
    return PageSection(
        key="ranges",
        title="Activation range of each cluster",
        lede_html=(
            "Each box is one cluster's hits, largest cluster first. The grey box is a random sample of all positions, "
            "for scale. A cluster that sits high fires the neuron hardest."
        ),
        panes=[Pane(name="Ranges", body_html=embed(cluster_ranges_figure(ranges, cluster_ids), "50vh"))],
    )


def cluster_ranges_figure(ranges: ActivationRanges, cluster_ids: list[int]) -> go.Figure:
    fig = go.Figure()
    shown = [c for c in cluster_ids if c in ranges.clusters]
    for cid in shown:
        fig.add_trace(go.Box(
            y=ranges.clusters[cid], name=tree_name(cid), marker_color=tree_color(cid), boxpoints="outliers",
            line_width=1, marker_size=3,
        ))
    if ranges.noise:
        fig.add_trace(go.Box(y=ranges.noise, name="all positions", marker_color=MUTED, boxpoints=False, line_width=1))
    values = np.concatenate([np.asarray(ranges.clusters[c]) for c in shown] + [np.asarray(ranges.noise)])
    log_axis = len(values) > 0 and bool((values > 0).all())
    fig.update_yaxes(title="activation" + (" (log scale)" if log_axis else ""), type="log" if log_axis else "linear")
    fig.update_layout(showlegend=False)
    return fig
