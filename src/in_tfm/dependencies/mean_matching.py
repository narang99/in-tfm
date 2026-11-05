"""Matching a writer firing to the writer's clusters when only their means were kept.

- Batch reports keep only each cluster's `mean.pt`, the mean Hadamard row of its hits.
  - There are no hit positions, so `matching.WriterMatcher` cannot rebuild the clustered points.
- A firing's own Hadamard row matches the mean nearest by cosine.
  - The distance is 1 - cosine, kept beside the match as in `matching`.
- The rows themselves are not kept on the match.
  - A big cluster has ~200k firings, so two rows each would make `firings.json` gigabytes.
  - The page shows the matched cluster's own `hadamard.jpg` instead.
"""

import re
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from .firings import ClusterMatch, WriterReportRef
from .matching import hadamard_rows
from .refs import Tap

CLUSTER_DIR = re.compile(r"cluster_(\d+)$")

ClusterMeans = dict[int, Float[np.ndarray, "hidden"]]


def load_cluster_means(report_dir: Path) -> ClusterMeans:
    return {
        int(match.group(1)): torch.load(path.parent / "mean.pt").double().numpy()
        for path in sorted(report_dir.glob("cluster_*/mean.pt"))
        if (match := CLUSTER_DIR.search(path.parent.name))
    }


def unit_rows(rows: Float[np.ndarray, "n hidden"]) -> Float[np.ndarray, "n hidden"]:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1, norms)


class MeanMatcher:
    def __init__(
        self, report: WriterReportRef, weight_row: Float[torch.Tensor, "hidden"]
    ) -> None:
        means = load_cluster_means(report.report_dir)
        if not means:
            raise ValueError(f"{report.report_dir} has no cluster_*/mean.pt")
        self.report = report
        self.weight_row = weight_row
        self.cluster_ids = list(means)
        self.means = np.stack([means[cid] for cid in self.cluster_ids])
        self.unit_means = unit_rows(self.means)

    @property
    def tap(self) -> Tap:
        return Tap(layer=self.report.layer, port="input")

    def match(self, inputs: Float[torch.Tensor, "n hidden"]) -> list[ClusterMatch]:
        own = hadamard_rows(inputs, self.weight_row)
        cosines = unit_rows(own) @ self.unit_means.T
        closest = cosines.argmax(axis=1)
        return [
            ClusterMatch(
                cluster_id=self.cluster_ids[ref],
                distance=float(1 - cosines[i, ref]),
                hadamard=[],
                closest_hadamard=[],
            )
            for i, ref in enumerate(closest)
        ]
