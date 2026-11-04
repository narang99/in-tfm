"""One card per frequent dependency, built from the stats through the view policy."""

from collections import defaultdict

from pydantic import BaseModel

from ..dependencies.collect import ClusterDependencies
from ..dependencies.firings import DependencyFiring
from ..dependencies.stats import (
    DependencyStats,
    DependencySummary,
    WriterSummary,
    writer_label,
)
from ..dependencies.view import TrimmedView, ViewPolicy, view
from .sparkline import Sparkline, sparkline


class DependencyCard(BaseModel):
    dim: int
    title: str
    summary: DependencySummary
    sparkline: Sparkline
    writers: TrimmedView[WriterSummary]

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
) -> DependencyCard:
    return DependencyCard(
        dim=summary.dim,
        title=dependency_title(summary.dim, firings),
        summary=summary,
        sparkline=sparkline(stats.concentration, summary.dim),
        writers=view(summary.writers, policy, lambda w: w.share, lambda w: w.label),
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
    collected: ClusterDependencies, stats: DependencyStats, policy: ViewPolicy
) -> DependencyCards:
    by_dim = firings_by_dim(collected.firings)
    frequent = [d for d in stats.dependencies if d.frequent]
    shown = view(frequent, policy, lambda d: d.share, lambda d: f"dim {d.dim}")
    return DependencyCards(
        cards=[
            dependency_card(row.members[0], by_dim[row.members[0].dim], stats, policy)
            for row in shown.shown
        ],
        rest=TrimmedView(shown=[], rest=shown.rest, description=shown.description),
        titles={
            d.dim: dependency_title(d.dim, by_dim[d.dim]) for d in stats.dependencies
        },
        one_offs=one_offs(stats.dependencies),
    )
