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
from ..report_page import PLOTLY_JS
from ..report_page.sections import StatTile
from .cards import DependencyCards, dependency_cards
from .figures import histogram_rows

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
) -> str:
    origin = collected.origin
    return render_template(
        "dependency_page.html",
        origin=origin,
        selection=collected.selection,
        settings=stats.settings,
        tiles=stat_tiles(collected, stats),
        histograms_html=histogram_rows(
            stats.distances, stats.fractions_above, stats.settings.threshold
        ),
        cards=cards,
    )


def render_dependency_page(
    collected: ClusterDependencies,
    settings: StatsSettings = StatsSettings(),
    policy: ViewPolicy = ViewPolicy(),
) -> str:
    stats = dependency_stats(collected, settings)
    origin = collected.origin
    title = f"{origin.layer}:{origin.neuron} cluster {origin.cluster_id} dependencies"
    body = page_body(collected, stats, dependency_cards(collected, stats, policy))
    return page(title, body, PLOTLY_JS)


def write_dependency_page(
    report_dir: Path,
    cluster_id: int,
    settings: StatsSettings = StatsSettings(),
    policy: ViewPolicy = ViewPolicy(),
) -> Path:
    collected = load_cluster_dependencies(report_dir, cluster_id)
    path = dependencies_dir(report_dir, cluster_id) / PAGE_NAME
    path.write_text(render_dependency_page(collected, settings, policy))
    return path
