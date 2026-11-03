"""The numbered sections of a neuron's report page.

- A section is a title, a lede, and one or more panes shown as tabs.
- Each section also contributes terms to the page's glossary.
- A builder returns None when its data is missing, so the page only numbers what it shows.
"""

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Int
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


def activations_section(curves: list[SelectionCurve]) -> PageSection | None:
    if not curves:
        return None
    panes = [
        Pane(name="Sorted curve", body_html=embed(sorted_curve_figure(curves), "55vh"))
    ]
    return PageSection(
        key="activations",
        title="Activations and the threshold",
        lede_html=one_tail_lede(curves[0]) if len(curves) == 1 else signed_lede(curves),
        panes=panes,
        guide=[
            GuideEntry(
                term="Hit",
                text="A position where the neuron fires past the threshold. Only hits are clustered.",
            )
        ],
    )


def one_tail_lede(curve: SelectionCurve) -> str:
    return (
        f"{curve.n_positions:,} positions fire in this tail. Sorted low to high, most sit in a long flat stretch "
        f"and a few shoot up at the end. Hits are the positions at or past <b>{curve.threshold:.4g}</b>."
    )


def signed_lede(curves: list[SelectionCurve]) -> str:
    cuts = " or ".join(
        f"{'below' if c.tail == 'negative' else 'above'} <b>{signed_value(c, c.threshold):.4g}</b>"
        for c in curves
    )
    return (
        "Both tails are hits, each picked on its own. The curve is every position the neuron fires on, "
        f"from the most negative to the most positive. Hits are the positions {cuts}. "
        "Each tail's share of the hits follows how many positions pass its threshold."
    )


class TailPlacement(BaseModel):
    """Where a tail's sorted magnitudes sit on the figure.

    - A lone tail is drawn as magnitudes, rising to the right.
    - In a signed figure the negative tail is mirrored to the left of the positive one, with its values negated."""

    offset: int
    step: int
    sign: float

    def x(self, index: Int[np.ndarray, "n"] | int) -> Int[np.ndarray, "n"] | int:
        return self.offset + self.step * index


def signed_value(curve: SelectionCurve, magnitude: float) -> float:
    return -magnitude if curve.tail == "negative" else magnitude


def tail_placements(curves: list[SelectionCurve]) -> list[TailPlacement]:
    if len(curves) == 1:
        return [TailPlacement(offset=0, step=1, sign=1.0)]
    n_negative = sum(c.n_positions for c in curves if c.tail == "negative")
    return [
        TailPlacement(offset=n_negative - 1, step=-1, sign=-1.0)
        if c.tail == "negative"
        else TailPlacement(offset=n_negative, step=1, sign=1.0)
        for c in curves
    ]


def sorted_curve_figure(curves: list[SelectionCurve]) -> go.Figure:
    fig = go.Figure()
    for curve, place in zip(curves, tail_placements(curves)):
        fig.add_trace(
            go.Scattergl(
                x=place.x(curve.index),
                y=place.sign * curve.magnitude,
                mode="lines",
                name=f"{curve.tail} tail",
                line=dict(width=2.5),
            )
        )
        if curve.elbow_idx is not None:
            add_elbow(fig, curve, place)
        if curve.bucket_edges is not None:
            add_bucket_edges(fig, curve, place)
    signed = len(curves) > 1
    fig.update_xaxes(
        title="sorted index (most negative to most positive)"
        if signed
        else "sorted index (low to high)"
    )
    fig.update_yaxes(title="activation" if signed else "activation magnitude")
    return fig


def add_elbow(fig: go.Figure, curve: SelectionCurve, place: TailPlacement) -> None:
    assert curve.elbow_idx is not None
    elbow_x = place.x(curve.elbow_idx)
    fig.add_vrect(
        x0=elbow_x,
        x1=place.x(curve.n_positions - 1),
        fillcolor=MARK,
        opacity=0.1,
        line_width=0,
    )
    fig.add_vline(
        x=elbow_x,
        line=dict(color=MARK, dash="dash"),
        annotation_text=f"threshold {place.sign * curve.threshold:.4g}",
    )


def add_bucket_edges(
    fig: go.Figure, curve: SelectionCurve, place: TailPlacement
) -> None:
    """Edges are magnitudes, so each sits at the first shown value at or above it. The last edge is the max."""
    assert curve.bucket_edges is not None
    for edge in curve.bucket_edges[:-1]:
        index = int(
            curve.index[
                min(np.searchsorted(curve.magnitude, edge), len(curve.index) - 1)
            ]
        )
        fig.add_vline(
            x=place.x(index),
            line=dict(color=MARK, dash="dash", width=1),
            annotation_text=f"{place.sign * edge:.3g}",
        )


def cluster_ranges_section(
    ranges: ActivationRanges | None, cluster_ids: list[int]
) -> PageSection | None:
    if ranges is None or not ranges.clusters:
        return None
    return PageSection(
        key="ranges",
        title="Activation range of each cluster",
        lede_html=(
            "Each box is one cluster's hits, largest cluster first. The grey box is a random sample of all positions, "
            "for scale. A cluster that sits high fires the neuron hardest."
        ),
        panes=[
            Pane(
                name="Ranges",
                body_html=embed(cluster_ranges_figure(ranges, cluster_ids), "50vh"),
            )
        ],
    )


def cluster_ranges_figure(
    ranges: ActivationRanges, cluster_ids: list[int]
) -> go.Figure:
    fig = go.Figure()
    shown = [c for c in cluster_ids if c in ranges.clusters]
    for cid in shown:
        fig.add_trace(
            go.Box(
                y=ranges.clusters[cid],
                name=tree_name(cid),
                marker_color=tree_color(cid),
                boxpoints="outliers",
                line_width=1,
                marker_size=3,
            )
        )
    if ranges.noise:
        fig.add_trace(
            go.Box(
                y=ranges.noise,
                name="all positions",
                marker_color=MUTED,
                boxpoints=False,
                line_width=1,
            )
        )
    values = np.concatenate(
        [np.asarray(ranges.clusters[c]) for c in shown] + [np.asarray(ranges.noise)]
    )
    log_axis = len(values) > 0 and bool((values > 0).all())
    fig.update_yaxes(
        title="activation" + (" (log scale)" if log_axis else ""),
        type="log" if log_axis else "linear",
    )
    fig.update_layout(showlegend=False)
    return fig
