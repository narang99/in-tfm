"""What a neuron's activation looks like per cluster, and against its noise.

- Nothing here runs the model, since pass 1 already holds the neuron's activation at every position.
  - A hit's activation is read from the scan at the position that made it a hit.
  - The noise is a seeded random sample of the scan's valid positions, hits or not.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from .activation_range_figure import save_activation_range_plot
from .neuron_capture import ScanResult

RANGES_NAME = "activation_ranges.json"
PLOT_NAME = "activation_ranges.jpg"
NOISE_SAMPLE_SIZE = 1000
PLOT_SEED = 0


class ActivationRanges(BaseModel):
    clusters: dict[int, list[float]]
    """Cluster id to the signed activation of each of its hits. HDBSCAN noise (-1) is not a cluster."""
    noise: list[float]


def neuron_activations(scan: ScanResult, neuron_idx: int) -> Float[np.ndarray, "batch seq"]:
    return scan.neuron_column(neuron_idx)[..., 0].numpy()


def hit_activations(
    scan: ScanResult, neuron_idx: int, batch_idx: Int[np.ndarray, "n_hits"], token_idx: Int[np.ndarray, "n_hits"]
) -> Float[np.ndarray, "n_hits"]:
    return neuron_activations(scan, neuron_idx)[batch_idx, token_idx]


def sample_noise(scan: ScanResult, neuron_idx: int, seed: int, size: int = NOISE_SAMPLE_SIZE) -> Float[np.ndarray, "n"]:
    valid_values = neuron_activations(scan, neuron_idx)[scan.valid_mask.numpy()]
    return np.random.default_rng(seed).choice(valid_values, size=min(size, len(valid_values)), replace=False)


def cluster_activations(
    labels: Int[np.ndarray, "n_hits"], activations: Float[np.ndarray, "n_hits"], cluster_ids: list[int]
) -> dict[int, list[float]]:
    return {cid: activations[labels == cid].tolist() for cid in cluster_ids}


def write_activation_ranges(report_dir: Path, ranges: ActivationRanges) -> None:
    (report_dir / RANGES_NAME).write_text(ranges.model_dump_json())
    if ranges.clusters:
        save_activation_range_plot(
            ranges.clusters, np.asarray(ranges.noise), report_dir / PLOT_NAME, np.random.default_rng(PLOT_SEED)
        )
