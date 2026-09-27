"""Matching new data against a cluster found during report generation, using cosine similarity
against the cluster's mean Hadamard vector rather than re-running HDBSCAN - HDBSCAN never sees
inference-time data, so it can't be the matching rule there.

The threshold itself is fit at report time from the cluster's own spread, needing no model
pass - see `neuron_report.NeuronClusterHits._inference_threshold`. What needs a pass is finding
out how far that threshold reaches across a corpus, which is `measure_corpus_admission`.
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


def measure_corpus_admission(
    report_dir: str | Path,
    cluster_ids: list[int],
    similarities: Float[torch.Tensor, "batch seq n_clusters"],
    valid_mask: Bool[torch.Tensor, "batch seq"],
    presenter: ClusterPresenter,
) -> dict[int, float]:
    """How much of a corpus each cluster's already-fitted threshold admits, plus the plot of the
    similarity distribution it was applied to.

    This is the diagnostic that catches a threshold which has drifted away from the concept it is
    supposed to name. A cluster holding 0.4% of tokens whose threshold admits 20% of them is not
    matching a concept, it is matching the bulk of the distribution.

    `cluster_ids` must be in the same order the similarity scan was given the cluster means in.
    """
    report_dir = Path(report_dir)
    meta = ReportMeta.model_validate_json((report_dir / "meta.json").read_text())
    admission: dict[int, float] = {}
    for i, cid in enumerate(cluster_ids):
        values = np.sort(similarities[..., i][valid_mask].numpy())
        threshold = meta.clusters[cid].inference_threshold
        admission[cid] = float((values >= threshold).mean())
        save_elbow_plot(
            values,
            int(np.searchsorted(values, threshold)),
            cluster_dir_for(report_dir, cid) / COSINE_ELBOW_PLOT_NAME,
            ylabel="cosine similarity",
        )
    render_report(report_dir, presenter)
    return admission
