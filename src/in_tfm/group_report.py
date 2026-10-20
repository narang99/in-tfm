"""The parts of a report page that show a grouping: its summary, its plots and one section per group.

- A report without a grouping shows none of this, and its page is unchanged.
- The plots are drawn from what the report folder stores, so a regrouped report gets them without the model.
  - The PCA uses the displayed hits and the cluster means, since the other hits' vectors are not stored.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel
from sklearn.decomposition import PCA

from .activation_ranges import read_activation_ranges
from .activation_ranges.figure import save_activation_range_plot
from .cluster_units import ClusterGroup, ClusterLike, activations_by_id, median_activation, unit_means
from .html_report import ACCENTS, accent_class, render_template
from .report_meta import GroupingMeta, KMeansGroupingMeta, KMeansSweep

KMEANS_PLOT_NAME = "kmeans_sweep.png"
GROUP_RANGES_PLOT_NAME = "group_activation_ranges.jpg"
PCA_PLOT_NAME = "group_pca.png"
NOISE_COLOR = "#9aa0a6"


class GroupView(BaseModel):
    id: int
    cluster_ids: list[int]
    n_hits: int
    n_unique_samples: int | None
    activation_min: float
    activation_max: float
    median_activation: float
    min_member_cosine: float


class PlotLink(BaseModel):
    src: str
    alt: str


class SimilarityRow(BaseModel):
    cluster_id: int
    values: list[float]


class SimilarityTable(BaseModel):
    cluster_ids: list[int]
    rows: list[SimilarityRow]


def group_view(group: ClusterGroup) -> GroupView:
    activations = group.activations()
    return GroupView(
        id=group.id,
        cluster_ids=group.cluster_ids,
        n_hits=group.n_hits,
        n_unique_samples=group.n_unique_samples,
        activation_min=float(activations.min()),
        activation_max=float(activations.max()),
        median_activation=median_activation(group),
        min_member_cosine=float(group.member_cosines().min()),
    )


def describe(name: str, grouping: GroupingMeta) -> str:
    if isinstance(grouping, KMeansGroupingMeta):
        how = f"k-means over the cluster means, k={grouping.k}" + (" from the elbow" if grouping.sweep else "")
    else:
        how = (
            f"cosine at least {grouping.min_cosine}, median activations within "
            f"{grouping.max_median_gap:.0%} of the largest"
        )
    return f"{len(grouping.members)} from {name} ({how})"


def similarity_table(units: Sequence[ClusterLike]) -> SimilarityTable:
    means = unit_means(units)
    similarity = (means @ means.T).tolist()
    return SimilarityTable(
        cluster_ids=[u.id for u in units],
        rows=[SimilarityRow(cluster_id=u.id, values=row) for u, row in zip(units, similarity)],
    )


def save_kmeans_sweep_plot(sweep: KMeansSweep, chosen_k: int, path: Path) -> None:
    fig = Figure(figsize=(10, 3.4))
    left, right = fig.subplots(1, 2)
    left.plot(sweep.ks, sweep.inertia, "o-")
    left.set_title("inertia")
    right.plot(list(sweep.silhouette), list(sweep.silhouette.values()), "o-", color="tab:green")
    right.set_title("silhouette")
    for ax in (left, right):
        ax.axvline(chosen_k, color="tab:orange", ls="--", label=f"k={chosen_k}")
        ax.set_xlabel("k")
    left.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def save_group_pca_plot(
    points: np.ndarray, point_groups: np.ndarray, groups: Sequence[ClusterGroup], path: Path
) -> None:
    """Hits as dots and cluster means as stars, both coloured by group, on the first two components."""
    pca = PCA(n_components=2).fit(points)
    xy = pca.transform(points)
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    for group in groups:
        color = ACCENTS[group.id % len(ACCENTS)]
        mask = point_groups == group.id
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c=color, alpha=0.6, linewidths=0, label=f"group {group.id}")
        stars = pca.transform(unit_means(group.clusters).numpy())
        ax.scatter(stars[:, 0], stars[:, 1], s=90, marker="*", c=color, edgecolors="black", linewidths=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("displayed hits (dots) and cluster means (stars)")
    if len(groups) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def group_plots(
    report_dir: Path, grouping: GroupingMeta, groups: Sequence[ClusterGroup], displayed: dict[int, np.ndarray]
) -> list[PlotLink]:
    """`displayed` is each cluster's displayed hits' Hadamard vectors, which the page loads anyway."""
    plots = []
    if isinstance(grouping, KMeansGroupingMeta) and grouping.sweep is not None:
        save_kmeans_sweep_plot(grouping.sweep, grouping.k, report_dir / KMEANS_PLOT_NAME)
        plots.append(PlotLink(src=KMEANS_PLOT_NAME, alt="k-means sweep"))
    ranges = read_activation_ranges(report_dir)
    noise = np.asarray(ranges.noise if ranges else [])
    save_activation_range_plot(activations_by_id(groups), noise, report_dir / GROUP_RANGES_PLOT_NAME, np.random.default_rng(0))
    plots.append(PlotLink(src=GROUP_RANGES_PLOT_NAME, alt="activation ranges per group"))
    points = [displayed[cid] for g in groups for cid in g.cluster_ids if len(displayed[cid])]
    if points and sum(len(p) for p in points) >= 3 and points[0].shape[1] >= 2:
        point_groups = np.concatenate([[g.id] * len(displayed[cid]) for g in groups for cid in g.cluster_ids])
        save_group_pca_plot(np.concatenate(points), point_groups, groups, report_dir / PCA_PLOT_NAME)
        plots.append(PlotLink(src=PCA_PLOT_NAME, alt="PCA of hits by group"))
    return plots


def groups_summary(
    name: str, grouping: GroupingMeta, groups: Sequence[ClusterGroup], plots: list[PlotLink]
) -> str:
    clusters_in_group_order = [c for g in groups for c in g.clusters]
    return render_template(
        "groups_summary.html",
        description=describe(name, grouping),
        groups=[group_view(g) for g in groups],
        plots=plots,
        similarity=similarity_table(clusters_in_group_order),
    )


def group_section(group: ClusterGroup, position: int, cluster_sections: Sequence[str]) -> str:
    return render_template(
        "group_section.html", group=group_view(group), accent_class=accent_class(position), body="\n".join(cluster_sections)
    )


def group_nav(groups: Sequence[ClusterGroup]) -> str:
    return render_template(
        "group_nav.html", groups=[group_view(g) for g in groups], accents=[accent_class(i) for i in range(len(groups))]
    )
