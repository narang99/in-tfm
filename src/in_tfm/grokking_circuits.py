"""Reading the clusters of every neuron as frequency groups, and groups of neurons as the subgraphs of a circuit."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from .grokking_fourier import N_FREQS, PairGrid, pair_frequency_energy
from .models.grokking.model import MODULUS, SEQ_LEN

EQUALS_POSITION = SEQ_LEN - 1


class ClusterFrequency(BaseModel):
    label: int
    n_hits: int
    dominant_freq: int
    share: float
    """Share of the hit-indicator's (a, b) energy at the dominant frequency, against a baseline of about 1 / 56 for no structure."""


class NeuronClusters(BaseModel):
    neuron: int
    n_hits: int
    n_noise: int
    clusters: list[ClusterFrequency]

    def frequency_votes(self) -> dict[int, int]:
        votes: dict[int, int] = {}
        for cluster in self.clusters:
            votes[cluster.dominant_freq] = votes.get(cluster.dominant_freq, 0) + cluster.n_hits
        return votes


def load_neuron_arrays(path: Path, polarity: str = "positive") -> dict[int, dict[str, np.ndarray]]:
    data = np.load(path)
    neurons = sorted({int(k.split("_")[0]) for k in data.files})
    return {
        n: {part: data[f"{n}_{polarity}_{part}"] for part in ("batch", "token", "label", "act")}
        for n in neurons
    }


def indicator_grid(batch_idx: Int[np.ndarray, "n"]) -> PairGrid:
    """Samples are all pairs in `a`-major order, so a sample index is `a * P + b`."""
    grid = np.zeros((MODULUS, MODULUS))
    grid[batch_idx // MODULUS, batch_idx % MODULUS] = 1.0
    return grid


def dominant_frequency(grid: PairGrid) -> tuple[int, float]:
    energy = pair_frequency_energy(grid)
    involving = energy[1:].sum(axis=1) + energy[:, 1:].sum(axis=0) - np.diag(energy)[1:]
    k = int(involving.argmax())
    return k + 1, float(involving[k] / energy.sum())


def cluster_frequencies(neuron: int, arrays: dict[str, np.ndarray], min_hits: int = 20) -> NeuronClusters:
    """Only the `=` position is read, since that is where the answer is formed from both inputs."""
    at_equals = arrays["token"] == EQUALS_POSITION
    labels = arrays["label"][at_equals]
    batch = arrays["batch"][at_equals]
    clusters = []
    for label in sorted(set(labels.tolist()) - {-1}):
        members = batch[labels == label]
        if len(members) < min_hits:
            continue
        freq, share = dominant_frequency(indicator_grid(members))
        clusters.append(ClusterFrequency(label=label, n_hits=len(members), dominant_freq=freq, share=share))
    return NeuronClusters(
        neuron=neuron, n_hits=len(labels), n_noise=int((labels == -1).sum()), clusters=clusters
    )


def hellinger_spectra(energy: Float[np.ndarray, "n_neurons freq"]) -> Float[np.ndarray, "n_neurons freq"]:
    """Unit-norm square roots of the energy shares, so Euclidean distance is the Hellinger distance between spectra."""
    shares = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
    return np.sqrt(shares)
