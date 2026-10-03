"""Matching new data against a cluster found during report generation, using cosine similarity
against the cluster's mean Hadamard vector rather than re-running HDBSCAN - HDBSCAN never sees
inference-time data, so it can't be the matching rule there.

The threshold that decides a match is fit separately from report writing, in
`fit_and_patch_inference_thresholds`, because fitting it needs a full model pass (see
neuron_report.py's module docstring for why report writing itself never makes one).
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float

from .hadamard import cosine_similarity
from .neuron_report import (
    COSINE_ELBOW_PLOT_NAME,
    MEAN_VECTOR_NAME,
    ReportMeta,
    cluster_dir_for,
    render_report,
)
from .presenters import ClusterPresenter
from .threshold import find_elbow_index_in_sorted_data
from .viz import save_elbow_plot


class ClusterMatcher:
    """A cluster's mean Hadamard vector plus its fitted cosine-similarity cutoff."""

    def __init__(self, mean_vector: Float[np.ndarray, "hidden"], threshold: float) -> None:
        self.mean_vector = mean_vector
        self.threshold = threshold

    @classmethod
    def from_report(cls, report_dir: str | Path, cluster_id: int) -> "ClusterMatcher":
        report_dir = Path(report_dir)
        meta = ReportMeta.model_validate_json((report_dir / "meta.json").read_text())
        threshold = meta.clusters[cluster_id].inference_threshold
        if threshold is None:
            raise ValueError(
                f"cluster {cluster_id} in {report_dir} has no inference threshold yet - run "
                "fit_and_patch_inference_thresholds first"
            )
        mean_vector = torch.load(
            cluster_dir_for(report_dir, cluster_id) / MEAN_VECTOR_NAME, weights_only=False
        ).numpy()
        return cls(mean_vector, threshold)

    def similarity(self, hdmds: Float[np.ndarray, "n hidden"]) -> Float[np.ndarray, "n"]:
        return cosine_similarity(hdmds, self.mean_vector)

    def is_hit(self, hdmds: Float[np.ndarray, "n hidden"]) -> np.ndarray:
        return self.similarity(hdmds) >= self.threshold


def load_cluster_means(
    report_dir: str | Path, cluster_ids: list[int]
) -> Float[np.ndarray, "n_clusters hidden"]:
    """Stacks the already-written mean vectors for `cluster_ids`, in that order - the order
    `fit_and_patch_inference_thresholds` needs to line up with a similarity scan's columns."""
    report_dir = Path(report_dir)
    return np.stack(
        [
            torch.load(cluster_dir_for(report_dir, cid) / MEAN_VECTOR_NAME, weights_only=False).numpy()
            for cid in cluster_ids
        ]
    )


def fit_and_patch_inference_thresholds(
    report_dir: str | Path,
    cluster_ids: list[int],
    similarities: Float[torch.Tensor, "batch seq n_clusters"],
    valid_mask: Bool[torch.Tensor, "batch seq"],
    presenter: ClusterPresenter,
) -> Path:
    """Elbow-fits each cluster's inference threshold against its column of a similarity scan
    (see `NeuronCapture.scan_cluster_similarities`) over every valid token in the corpus, writes
    the cosine-similarity elbow plot, patches `meta.json`, and re-renders the report so the
    plots and thresholds show up in `index.html`.

    `cluster_ids` must be in the same order the similarity scan was given the cluster means in.
    """
    report_dir = Path(report_dir)
    meta = ReportMeta.model_validate_json((report_dir / "meta.json").read_text())
    for i, cid in enumerate(cluster_ids):
        values = np.sort(similarities[..., i][valid_mask].numpy())
        elbow_idx = find_elbow_index_in_sorted_data(torch.from_numpy(values))
        cluster_dir = cluster_dir_for(report_dir, cid)
        save_elbow_plot(
            values, elbow_idx, cluster_dir / COSINE_ELBOW_PLOT_NAME, ylabel="cosine similarity"
        )
        meta.clusters[cid].inference_threshold = float(values[elbow_idx])
    (report_dir / "meta.json").write_text(meta.model_dump_json(indent=2))
    return render_report(report_dir, presenter)
