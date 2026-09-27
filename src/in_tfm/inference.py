"""Matching new data against a cluster found during report generation, using cosine similarity
against the cluster's mean Hadamard vector rather than re-running HDBSCAN - HDBSCAN never sees
inference-time data, so it can't be the matching rule there.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from .hadamard import cosine_similarity
from .neuron_report import MEAN_VECTOR_NAME, ReportMeta, cluster_dir_for


class ClusterMatcher:
    """A cluster's mean Hadamard vector plus the cosine-similarity cutoff fit against every
    valid token in the data that produced it - see NeuronClusterHits._inference_threshold."""

    def __init__(self, mean_vector: Float[np.ndarray, "hidden"], threshold: float) -> None:
        self.mean_vector = mean_vector
        self.threshold = threshold

    @classmethod
    def from_report(cls, report_dir: str | Path, cluster_id: int) -> "ClusterMatcher":
        report_dir = Path(report_dir)
        meta = ReportMeta.model_validate_json((report_dir / "meta.json").read_text())
        mean_vector = torch.load(
            cluster_dir_for(report_dir, cluster_id) / MEAN_VECTOR_NAME, weights_only=False
        ).numpy()
        return cls(mean_vector, meta.clusters[cluster_id].inference_threshold)

    def similarity(self, hdmds: Float[np.ndarray, "n hidden"]) -> Float[np.ndarray, "n"]:
        return cosine_similarity(hdmds, self.mean_vector)

    def is_hit(self, hdmds: Float[np.ndarray, "n hidden"]) -> np.ndarray:
        return self.similarity(hdmds) >= self.threshold
