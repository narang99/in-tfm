"""Groups a report's clusters, and stores the result as a named grouping in its `meta.json`."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Int

from ..cluster_units import ClusterLike, load_clusters
from ..report_meta import GroupingMeta, KMeansGroupingMeta, ThresholdGroupingMeta, load_report_meta, write_report_meta
from ..run_config import MergeClustersConfig
from .strategies import kmeans_assignment, threshold_assignment

MIN_UNITS_FOR_KMEANS = 3
"""The k sweep needs k from 1 to one less than the unit count, and silhouette needs k of 2 or more."""


def members_from_assignment(units: Sequence[ClusterLike], assignment: Int[np.ndarray, "units"]) -> list[list[int]]:
    """Largest group first by hits. Inside a group, units keep their input order."""
    members: dict[int, list[ClusterLike]] = {}
    for unit, label in zip(units, assignment):
        members.setdefault(int(label), []).append(unit)
    ordered = sorted(members.values(), key=lambda group: -sum(u.n_hits for u in group))
    return [[u.id for u in group] for group in ordered]


def merge_clusters(units: Sequence[ClusterLike], config: MergeClustersConfig, seed: int) -> GroupingMeta:
    if config.method == "kmeans":
        if len(units) < MIN_UNITS_FOR_KMEANS:
            return KMeansGroupingMeta(k=len(units), sweep=None, members=[[u.id] for u in units])
        assignment, k, sweep = kmeans_assignment(units, config.k, seed)
        return KMeansGroupingMeta(k=k, sweep=sweep, members=members_from_assignment(units, assignment))
    if config.method == "thresholds":
        assignment = threshold_assignment(units, config.min_cosine, config.max_median_gap)
        return ThresholdGroupingMeta(
            min_cosine=config.min_cosine,
            max_median_gap=config.max_median_gap,
            members=members_from_assignment(units, assignment),
        )
    raise ValueError("method none has nothing to merge")


def grouping_name(config: MergeClustersConfig) -> str:
    """Names the settings, so regrouping with other settings adds a grouping instead of replacing one."""
    if config.method == "kmeans":
        return "kmeans" if config.k is None else f"kmeans-k{config.k}"
    return f"thresholds-cos{config.min_cosine}-gap{config.max_median_gap}"


def merge_report_clusters(report_dir: Path, config: MergeClustersConfig, seed: int) -> str | None:
    """Adds the grouping to the report's `meta.json` and makes it the one the page shows.

    None for method none, and for a report without clusters, which has nothing to group.
    """
    if config.method == "none":
        return None
    clusters = load_clusters(report_dir)
    if not clusters:
        return None
    meta = load_report_meta(report_dir)
    name = grouping_name(config)
    grouping = merge_clusters(clusters, config, seed)
    write_report_meta(report_dir, meta.model_copy(update={"groupings": {**meta.groupings, name: grouping}, "default_grouping": name}))
    return name
