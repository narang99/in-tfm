"""Reads a report folder back as clusters, the one place that knows the folder's file layout."""

from pathlib import Path

import numpy as np
import torch

from ..activation_ranges import read_activation_ranges
from ..report_meta import load_report_meta
from .hit_table import read_hit_table
from .units import Cluster, ClusterGroup, singleton_groups

MEAN_NAME = "mean.pt"


def cluster_dir_for(report_dir: Path, cluster_id: int) -> Path:
    return report_dir / f"cluster_{cluster_id}"


def load_clusters(report_dir: Path) -> list[Cluster]:
    """In the report's order, largest first."""
    meta = load_report_meta(report_dir)
    ranges = read_activation_ranges(report_dir)
    return [
        Cluster(
            id=cid,
            n_hits=cluster.n_hits,
            n_unique_samples=cluster.n_unique_images,
            mean=torch.load(cluster_dir_for(report_dir, cid) / MEAN_NAME),
            hit_activations=None if ranges is None else np.asarray(ranges.clusters[cid]),
            cluster_dir=cluster_dir_for(report_dir, cid),
        )
        for cid, cluster in meta.clusters.items()
    ]


def load_singleton_groups(report_dir: Path) -> list[ClusterGroup]:
    return singleton_groups(load_clusters(report_dir), read_hit_table(report_dir))
