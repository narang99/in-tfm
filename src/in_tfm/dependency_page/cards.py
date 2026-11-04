"""One card per frequent dependency, built from the stats through the view policy."""

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from ..dependencies.collect import ClusterDependencies, dependencies_dir
from ..dependencies.firings import DependencyFiring
from ..dependencies.stats import (
    DependencyStats,
    DependencySummary,
    WriterSummary,
    writer_label,
)
from ..dependencies.view import TrimmedView, ViewPolicy, view
from ..presenters import ClusterPresenter
from .clusters import (
    ReportsByWriter,
    WriterClusters,
    reports_by_writer,
    writer_clusters,
)
from .sparkline import Sparkline, sparkline


class DependencyCard(BaseModel):
    dim: int
    title: str
    summary: DependencySummary
    sparkline: Sparkline
    writers: TrimmedView[WriterSummary]
    clusters: list[WriterClusters] = []
    """Phase B, for each shown writer matched to its own clusters."""

    @property
    def anchor(self) -> str:
        return f"dim-{self.dim}"


def firings_by_dim(
    firings: list[DependencyFiring],
) -> dict[int, list[DependencyFiring]]:
    grouped: dict[int, list[DependencyFiring]] = defaultdict(list)
    for firing in firings:
        grouped[firing.dim].append(firing)
    return grouped


def dependency_title(dim: int, firings: list[DependencyFiring]) -> str:
    """A dim fed by one writer is named after it. A sum of writers, like a stream dim, by its dim."""
    first = firings[0]
    if first.n_writers == 1:
        return f"{writer_label(first.writers[0].writer)} at token {first.writers[0].writer.token}"
    return f"dim {dim}"


def dependency_card(
    summary: DependencySummary,
    firings: list[DependencyFiring],
    stats: DependencyStats,
    policy: ViewPolicy,
    reports: ReportsByWriter,
    page_dir: Path,
    presenter: ClusterPresenter | None,
) -> DependencyCard:
    writers = view(summary.writers, policy, lambda w: w.share, lambda w: w.label)
    shown = [row.members[0] for row in writers.shown]
    matched = [writer_clusters(w, reports, page_dir, presenter) for w in shown]
    return DependencyCard(
        dim=summary.dim,
        title=dependency_title(summary.dim, firings),
        summary=summary,
        sparkline=sparkline(stats.concentration, summary.dim),
        writers=writers,
        clusters=[m for m in matched if m is not None],
    )


class DependencyCards(BaseModel):
    cards: list[DependencyCard]
    rest: TrimmedView[DependencySummary]
    """The frequent dependencies the policy did not show. `shown` is empty."""
    titles: dict[int, str]
    one_offs: list[DependencySummary]


def one_offs(dependencies: list[DependencySummary]) -> list[DependencySummary]:
    rare = [d for d in dependencies if not d.frequent]
    return sorted(rare, key=lambda d: -d.share)


def dependency_cards(
    collected: ClusterDependencies,
    stats: DependencyStats,
    policy: ViewPolicy,
    presenter: ClusterPresenter | None = None,
) -> DependencyCards:
    """Without a presenter, the cluster bars do not open to their hits."""
    by_dim = firings_by_dim(collected.firings)
    frequent = [d for d in stats.dependencies if d.frequent]
    shown = view(frequent, policy, lambda d: d.share, lambda d: f"dim {d.dim}")
    reports = reports_by_writer(collected.writer_reports)
    page_dir = dependencies_dir(
        collected.origin.report_dir, collected.origin.cluster_id
    )
    return DependencyCards(
        cards=[
            dependency_card(
                row.members[0],
                by_dim[row.members[0].dim],
                stats,
                policy,
                reports,
                page_dir,
                presenter,
            )
            for row in shown.shown
        ],
        rest=TrimmedView(shown=[], rest=shown.rest, description=shown.description),
        titles={
            d.dim: dependency_title(d.dim, by_dim[d.dim]) for d in stats.dependencies
        },
        one_offs=one_offs(stats.dependencies),
    )
