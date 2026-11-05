"""Phase B of a card: which of a writer's own clusters its firings matched.

- Each cluster links to its section in the writer's own report, rather than copying its hits.
- A cluster under the outlier ratio of the writer's matches is folded into one row, as hp-do did.
- Pairs are a seeded random pick of a cluster's firings, not its closest ones, so they show the spread.
"""

import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ..cluster_colors import cluster_style
from ..dependencies.firings import DependencyFiring, WriterReportRef
from ..dependencies.stats import ClusterShare, WriterSummary
from .tiles import pair_uri

PAIRS_PER_CLUSTER = 3


class ClusterRow(BaseModel):
    share: ClusterShare
    href: str
    style: str


class MatchPair(BaseModel):
    cluster_id: int
    style: str
    distance: float
    image_uri: str


class WriterClusters(BaseModel):
    label: str
    rows: list[ClusterRow]
    outliers: list[ClusterRow]
    n_matched: int
    pairs: list[MatchPair]

    @property
    def outlier_count(self) -> int:
        return sum(row.share.count for row in self.outliers)


ReportsByWriter = dict[tuple[str, int], WriterReportRef]


def reports_by_writer(reports: list[WriterReportRef]) -> ReportsByWriter:
    return {(r.layer, r.neuron): r for r in reports}


def section_href(report: WriterReportRef, cluster_id: int, page_dir: Path) -> str:
    index = os.path.relpath(report.report_dir / "index.html", page_dir)
    return f"{index}#cluster-{cluster_id}"


def cluster_row(
    share: ClusterShare, report: WriterReportRef, page_dir: Path
) -> ClusterRow:
    return ClusterRow(
        share=share,
        href=section_href(report, share.cluster_id, page_dir),
        style=cluster_style(share.cluster_id),
    )


def matched_pairs(
    firings: list[DependencyFiring],
    writer: WriterSummary,
    cluster_id: int,
    report: WriterReportRef,
) -> list[MatchPair]:
    matches = [
        w.match
        for f in firings
        for w in f.writers
        if w.writer.column == writer.column
        and w.match is not None
        and w.match.cluster_id == cluster_id
        and w.match.hadamard
    ]
    rng = np.random.default_rng(cluster_id)
    picked = rng.permutation(len(matches))[:PAIRS_PER_CLUSTER]
    return [
        MatchPair(
            cluster_id=cluster_id,
            style=cluster_style(cluster_id),
            distance=matches[i].distance,
            image_uri=pair_uri(
                matches[i].hadamard, matches[i].closest_hadamard, report.tile_shape
            ),
        )
        for i in picked
    ]


def writer_clusters(
    writer: WriterSummary,
    firings: list[DependencyFiring],
    reports: ReportsByWriter,
    page_dir: Path,
) -> WriterClusters | None:
    report = reports.get((writer.column.tap.layer, writer.column.column))
    if report is None or not writer.clusters:
        return None
    kept = [c for c in writer.clusters if not c.is_outlier]
    return WriterClusters(
        label=writer.label,
        rows=[cluster_row(c, report, page_dir) for c in kept],
        outliers=[
            cluster_row(c, report, page_dir) for c in writer.clusters if c.is_outlier
        ],
        n_matched=sum(c.count for c in writer.clusters),
        pairs=[
            p
            for c in kept
            for p in matched_pairs(firings, writer, c.cluster_id, report)
        ],
    )
