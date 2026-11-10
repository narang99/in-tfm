"""A cluster's dependency page, written next to its `firings.json`.

- It renders from the saved firings alone, so changing the stats settings or the view needs no model.
"""

from pathlib import Path

from ..dependencies.collect import (
    ClusterDependencies,
    dependencies_dir,
    load_cluster_dependencies,
)
from ..dependencies.stats import DependencyStats, StatsSettings, dependency_stats
from ..dependencies.view import ViewPolicy
from ..html_report import page, render_template
from ..presenters import ClusterPresenter
from ..report_page.sections import StatTile
from .cards import DependencyCards, dependency_cards

PAGE_NAME = "index.html"


def stat_tiles(
    collected: ClusterDependencies, stats: DependencyStats
) -> list[StatTile]:
    frequent = sum(d.frequent for d in stats.dependencies)
    return [
        StatTile(value=f"{stats.n_examples:,}", name="examples"),
        StatTile(value=f"{len(collected.firings):,}", name="firings"),
        StatTile(value=f"{len(stats.dependencies):,}", name="dependencies that fired"),
        StatTile(
            value=f"{frequent:,}",
            name=f"frequent (≥ {stats.frequent_threshold} examples)",
        ),
        StatTile(value=f"{len(stats.dependencies) - frequent:,}", name="one-off"),
    ]


def page_body(
    collected: ClusterDependencies,
    stats: DependencyStats,
    cards: DependencyCards,
    controls: str,
) -> str:
    origin = collected.origin
    return render_template(
        "dependency_page.html",
        origin=origin,
        tiles=stat_tiles(collected, stats),
        cards=cards,
        controls=controls,
    )


def render_dependency_page(
    collected: ClusterDependencies,
    settings: StatsSettings = StatsSettings(),
    policy: ViewPolicy = ViewPolicy(),
    presenter: ClusterPresenter | None = None,
) -> str:
    stats = dependency_stats(collected, settings)
    origin = collected.origin
    title = f"{origin.layer}:{origin.neuron} cluster {origin.cluster_id} dependencies"
    cards = dependency_cards(collected, stats, policy, presenter)
    controls = presenter.page_controls() if presenter else ""
    return page(title, page_body(collected, stats, cards, controls))


def write_dependency_page(
    report_dir: Path,
    cluster_id: int,
    settings: StatsSettings = StatsSettings(),
    policy: ViewPolicy = ViewPolicy(),
    presenter: ClusterPresenter | None = None,
) -> Path:
    collected = load_cluster_dependencies(report_dir, cluster_id)
    path = dependencies_dir(report_dir, cluster_id) / PAGE_NAME
    path.write_text(render_dependency_page(collected, settings, policy, presenter))
    return path
