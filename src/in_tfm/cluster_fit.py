"""A neuron's density-peak clusters, saved so a later trace can read cluster probabilities back.

- Every hit is stored by its corpus position, so a trace over the same corpus looks a position up.
- Each distinct point's geodesic distance to every tree's peak is kept too.
  - Assigning an unseen point later only needs its nearest stored hits, these distances and `tau`.
- The clustered rows are not stored.
  - They are the layer input times the neuron's weight row, so the corpus and model give them back.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from .density_peaks import DensityPeakForest, ForestTrees
from .density_peaks.probabilities import distances_to_peaks, softmax_over_trees, step_length
from .hit_selection import HitSelection
from .run_config import ClusteringConfig, DensityPeaksConfig, Polarity
from .sources import SampleId

FIT_META = "clusters.json"
FIT_ARRAYS = "clusters.npz"


class ClusterFitMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neuron_idx: int
    polarity: Polarity
    sample_ids: list[SampleId]
    """The scan's samples. A hit's `batch_idx` indexes into these."""
    distance: str
    normalise: bool
    k: int
    peak_share: float
    tau: float
    """The softmax temperature over geodesic distances, the median nearest-neighbour distance."""
    n_trees: int


class ClusterFit(BaseModel):
    """Per hit unless named per distinct point. Tree t is the report's cluster t."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: ClusterFitMeta
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    activations: Float[np.ndarray, "n_hits"]
    distinct_of_hit: Int[np.ndarray, "n_hits"]
    tree: Int[np.ndarray, "n_distinct"]
    kth_distance: Float[np.ndarray, "n_distinct"]
    peak_distances: Float[np.ndarray, "n_distinct n_trees"]
    """Geodesic, along the kNN graph. Inf when a tree's peak is out of reach."""
    probabilities: Float[np.ndarray, "n_distinct n_trees"]
    """As the report used them, including the own-tree fallback for a point that reaches no peak."""

    def hit_probabilities(self) -> Float[np.ndarray, "n_hits n_trees"]:
        return self.probabilities[self.distinct_of_hit]

    def probabilities_at(self, batch_idx: int, token_idx: int) -> Float[np.ndarray, "n_trees"]:
        """Raises `KeyError` for a position that was not a hit, rather than guessing a membership."""
        match = np.flatnonzero((self.batch_idx == batch_idx) & (self.token_idx == token_idx))
        if len(match) == 0:
            raise KeyError(f"position ({batch_idx}, {token_idx}) is not a hit of neuron {self.meta.neuron_idx}")
        return self.probabilities[self.distinct_of_hit[match[0]]]


def cluster_fit(
    forest: DensityPeakForest,
    trees: ForestTrees,
    probabilities: Float[np.ndarray, "n_distinct n_trees"],
    selection: HitSelection,
    sample_ids: list[SampleId],
    neuron_idx: int,
    clustering: ClusteringConfig,
    strategy: DensityPeaksConfig,
) -> ClusterFit:
    tau = step_length(forest)
    peak_distances = distances_to_peaks(forest, trees, strategy.peak_share)
    reachable = np.isfinite(peak_distances).any(axis=1)
    if not np.allclose(softmax_over_trees(peak_distances, tau)[reachable], probabilities[reachable]):
        raise ValueError("the stored distances do not give back the report's probabilities")
    return ClusterFit(
        meta=ClusterFitMeta(
            neuron_idx=neuron_idx, polarity=selection.polarity, sample_ids=list(sample_ids),
            distance=clustering.distance, normalise=clustering.normalise, k=strategy.k,
            peak_share=strategy.peak_share, tau=tau, n_trees=len(trees.roots),
        ),
        batch_idx=selection.batch_idx,
        token_idx=selection.token_idx,
        activations=selection.activations,
        distinct_of_hit=forest.distinct_of_hit,
        tree=trees.tree,
        kth_distance=forest.kth_distance,
        peak_distances=peak_distances,
        probabilities=probabilities,
    )


ARRAY_FIELDS = (
    "batch_idx", "token_idx", "activations", "distinct_of_hit", "tree", "kth_distance", "peak_distances",
    "probabilities",
)


def save_cluster_fit(fit: ClusterFit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FIT_META).write_text(fit.meta.model_dump_json(indent=2))
    np.savez_compressed(out_dir / FIT_ARRAYS, **{name: getattr(fit, name) for name in ARRAY_FIELDS})
    return out_dir / FIT_ARRAYS


def load_cluster_fit(out_dir: Path) -> ClusterFit:
    meta = ClusterFitMeta.model_validate_json((out_dir / FIT_META).read_text())
    with np.load(out_dir / FIT_ARRAYS) as arrays:
        return ClusterFit(meta=meta, **{name: arrays[name] for name in ARRAY_FIELDS})
