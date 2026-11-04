"""Phase B of a card: which of a writer's own clusters its firings matched, one bar each.

- A cluster under the outlier ratio of the writer's matches is folded under the others, as hp-do did.
- A bar opens to that cluster's hits, drawn by the writer report's presenter from its `hits.pt`.
  - The presenter writes its files into the page's own folder, so the page never reads the writer's html.
  - The bodies ship inside `<template>`, and only become DOM when a bar is opened.
"""

from pathlib import Path

from pydantic import BaseModel

from ..cluster_colors import cluster_style
from ..dependencies.firings import WriterReportRef
from ..dependencies.stats import ClusterShare, WriterSummary
from ..neuron_report import cluster_dir_for, load_hits
from ..presenters import ClusterArtifacts, ClusterPresenter
from ..report_meta import load_report_meta


class ClusterBar(BaseModel):
    share: ClusterShare
    style: str
    body: str | None
    """The cluster's hits as its own report draws them. None when the page has no presenter."""

    @property
    def percent(self) -> float:
        return 100 * self.share.ratio


class WriterClusters(BaseModel):
    label: str
    bars: list[ClusterBar]
    outliers: list[ClusterBar]


ReportsByWriter = dict[tuple[str, int], WriterReportRef]


def reports_by_writer(reports: list[WriterReportRef]) -> ReportsByWriter:
    return {(r.layer, r.neuron): r for r in reports}


def artifacts_dir(report: WriterReportRef, cluster_id: int, page_dir: Path) -> Path:
    """A direct child of the page's folder, since presenters link their files relative to it."""
    return page_dir / f"{report.layer}_{report.neuron}_cluster_{cluster_id}"


def cluster_body(
    report: WriterReportRef,
    cluster_id: int,
    page_dir: Path,
    presenter: ClusterPresenter,
) -> str:
    hits = load_hits(cluster_dir_for(report.report_dir, cluster_id))
    if not hits:
        return "<p>no hits sampled.</p>"
    out_dir = artifacts_dir(report, cluster_id, page_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hadamard_shape = load_report_meta(report.report_dir).hadamard_shape
    return presenter.render(hits, ClusterArtifacts(cluster_dir=out_dir), hadamard_shape)


def cluster_bar(
    share: ClusterShare,
    report: WriterReportRef,
    page_dir: Path,
    presenter: ClusterPresenter | None,
) -> ClusterBar:
    return ClusterBar(
        share=share,
        style=cluster_style(share.cluster_id),
        body=None
        if presenter is None
        else cluster_body(report, share.cluster_id, page_dir, presenter),
    )


def writer_clusters(
    writer: WriterSummary,
    reports: ReportsByWriter,
    page_dir: Path,
    presenter: ClusterPresenter | None,
) -> WriterClusters | None:
    report = reports.get((writer.column.tap.layer, writer.column.column))
    if report is None or not writer.clusters:
        return None
    return WriterClusters(
        label=writer.label,
        bars=[
            cluster_bar(c, report, page_dir, presenter)
            for c in writer.clusters
            if not c.is_outlier
        ],
        outliers=[
            cluster_bar(c, report, page_dir, presenter)
            for c in writer.clusters
            if c.is_outlier
        ],
    )
