"""One page for a whole run: every origin cluster, and the writer clusters its firings matched.

- An edge runs from an origin cluster to one writer neuron's cluster.
  - Its share is the part of the origin cluster's matched |contribution| that went through it.
  - So an origin cluster's edges sum to 1, and writers without a report are left out of the total.
- Edges go through the same view policy as the cards, so a wide graph is cut the same visible way.
"""

import os
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from pydantic import BaseModel

from ..cluster_colors import cluster_color, cluster_style
from ..dependencies.collect import ClusterDependencies, dependencies_dir
from ..dependencies.firings import writer_contribution
from ..dependencies.refs import NeuronWriter
from ..dependencies.view import TrimmedView, ViewPolicy, view
from ..html_report import page, render_template
from ..report_page import PLOTLY_JS
from ..report_page.figures import embed

GRAPH_PAGE = "dependency_graph.html"
ADDING_EDGE = "rgba(0,165,80,0.4)"
INHIBITING_EDGE = "rgba(221,51,51,0.4)"


class WriterCluster(BaseModel, frozen=True):
    layer: str
    neuron: int
    cluster_id: int

    @property
    def label(self) -> str:
        return f"{self.layer}:{self.neuron} c{self.cluster_id}"


class GraphEdge(BaseModel):
    target: WriterCluster
    share: float
    signed_contribution: float
    n_firings: int
    href: str
    style: str


class OriginNode(BaseModel):
    label: str
    href: str
    n_examples: int
    edges: TrimmedView[GraphEdge]


def origin_label(collected: ClusterDependencies) -> str:
    origin = collected.origin
    return f"{origin.layer}:{origin.neuron} c{origin.cluster_id}"


def writer_section_href(
    collected: ClusterDependencies, target: WriterCluster, from_dir: Path
) -> str:
    reports = {(r.layer, r.neuron): r for r in collected.writer_reports}
    report = reports[(target.layer, target.neuron)]
    index = os.path.relpath(report.report_dir / "index.html", from_dir)
    return f"{index}#cluster-{target.cluster_id}"


def origin_edges(collected: ClusterDependencies, from_dir: Path) -> list[GraphEdge]:
    magnitude: dict[WriterCluster, float] = defaultdict(float)
    signed: dict[WriterCluster, float] = defaultdict(float)
    count: dict[WriterCluster, int] = defaultdict(int)
    for firing in collected.firings:
        for writer in firing.writers:
            if writer.match is None or not isinstance(writer.writer, NeuronWriter):
                continue
            target = WriterCluster(
                layer=writer.writer.layer,
                neuron=writer.writer.neuron,
                cluster_id=writer.match.cluster_id,
            )
            contribution = writer_contribution(firing, writer)
            magnitude[target] += abs(contribution)
            signed[target] += contribution
            count[target] += 1
    total = sum(magnitude.values()) or 1.0
    return [
        GraphEdge(
            target=target,
            share=magnitude[target] / total,
            signed_contribution=signed[target],
            n_firings=count[target],
            href=writer_section_href(collected, target, from_dir),
            style=cluster_style(target.cluster_id),
        )
        for target in magnitude
    ]


def origin_node(
    collected: ClusterDependencies, out_dir: Path, policy: ViewPolicy
) -> OriginNode:
    page_dir = dependencies_dir(
        collected.origin.report_dir, collected.origin.cluster_id
    )
    return OriginNode(
        label=origin_label(collected),
        href=os.path.relpath(page_dir / "index.html", out_dir),
        n_examples=len(collected.examples),
        edges=view(
            origin_edges(collected, out_dir),
            policy,
            lambda e: e.share,
            lambda e: e.target.label,
        ),
    )


def sankey(nodes: list[OriginNode]) -> str:
    """Origin clusters on the left, writer clusters on the right, only the edges the view shows."""
    targets = sorted(
        {row.members[0].target for node in nodes for row in node.edges.shown},
        key=lambda t: (t.layer, t.neuron, t.cluster_id),
    )
    index = {t: len(nodes) + i for i, t in enumerate(targets)}
    links = [
        (i, index[edge.target], edge)
        for i, node in enumerate(nodes)
        for edge in (row.members[0] for row in node.edges.shown)
    ]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=[n.label for n in nodes] + [t.label for t in targets],
                color=["#9aa1ae"] * len(nodes)
                + [cluster_color(t.cluster_id) for t in targets],
                pad=6,
                thickness=12,
            ),
            link=dict(
                source=[s for s, _, _ in links],
                target=[t for _, t, _ in links],
                value=[e.share for _, _, e in links],
                color=[
                    ADDING_EDGE if e.signed_contribution > 0 else INHIBITING_EDGE
                    for _, _, e in links
                ],
                customdata=[e.n_firings for _, _, e in links],
                hovertemplate="%{source.label} → %{target.label}<br>%{value:.0%} of the origin's matched |contribution|<br>%{customdata} firings<extra></extra>",
            ),
        )
    )
    fig.update_layout(template="none")
    return embed(fig, f"{max(400, 22 * (len(nodes) + len(targets)))}px")


def graph_html(
    collections: list[ClusterDependencies], out_dir: Path, policy: ViewPolicy
) -> str:
    nodes = [origin_node(c, out_dir, policy) for c in collections]
    body = render_template(
        "dependency_graph.html",
        nodes=nodes,
        sankey_html=sankey(nodes),
        description=policy.describe(),
    )
    return page("Dependency graph", body, PLOTLY_JS)


def write_graph_page(
    collections: list[ClusterDependencies], out_dir: Path, policy: ViewPolicy
) -> Path:
    path = out_dir / GRAPH_PAGE
    path.write_text(graph_html(collections, out_dir, policy))
    return path
