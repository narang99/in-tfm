"""A neuron's hits saved once, so every report built later reads arrays and never re-clusters.

- Collecting is the slow part: the model scan, the full distance matrix and the density-peak forest.
- Every report built from one space shares its cluster ids, so the reports can link to each other.
- The distance matrix is the only large file, so it is loaded on its own and memory-mapped.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

from .density_levels import DensityGraph
from .density_peaks import DensityPeakForest, ForestTrees
from .hit_selection import HitSelection
from .hit_selection.picks import Pick
from .run_config import Polarity
from .sources import SampleId

HADAMARDS = "hadamards.npy"
HITS = "hits.npz"
SELECTION = "selection.json"
PICK_ARRAYS = "pick_arrays.npz"
SORTED_MAGNITUDES = "sorted_magnitudes.npy"
NOISE = "noise.npy"
DISTANCES = "distances.npy"
FOREST = "forest.npz"
TREES = "trees.npz"
PROBABILITIES = "probabilities.npy"
SPACE_FILES = (HADAMARDS, HITS, SELECTION, PICK_ARRAYS, SORTED_MAGNITUDES, NOISE, DISTANCES, FOREST, TREES, PROBABILITIES)


class SelectionRecord(BaseModel):
    """What the elbow or bucket plot needs from the selection, beside the hit positions in `hits.npz`.

    - A bucketed pick holds numpy arrays, which JSON cannot, so those go to `pick_arrays.npz`.
    """

    polarity: Polarity
    n_candidates: int
    pick_fields: dict[str, Any]
    """The pick without its arrays."""


def pick_arrays(pick: Pick) -> dict[str, np.ndarray]:
    return {name: value for name, value in vars(pick).items() if isinstance(value, np.ndarray)}


def load_pick(record: SelectionRecord, arrays: dict[str, np.ndarray]) -> Pick:
    return TypeAdapter(Pick).validate_python({**record.pick_fields, **arrays})


class HitSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hadamards: Float[np.ndarray, "n_hits hidden"]
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    activations: Float[np.ndarray, "n_hits"]
    sample_ids: list[SampleId]
    selection: SelectionRecord
    pick: Pick
    sorted_magnitudes: Float[np.ndarray, "n_pos"]
    noise: Float[np.ndarray, "n_noise"]
    """Activations at random valid positions, hits or not, for the activation range plot."""
    probabilities: Float[np.ndarray, "n_hits n_clusters"]
    """Per hit, read through the forest's `distinct_of_hit`. Cluster i is tree i, largest tree first."""
    distinct_of_hit: Int[np.ndarray, "n_hits"]
    roots: Int[np.ndarray, "n_clusters"]

    @property
    def n_clusters(self) -> int:
        return self.probabilities.shape[1]

    def hit_selection(self) -> HitSelection:
        return HitSelection(
            polarity=self.selection.polarity, batch_idx=self.batch_idx, token_idx=self.token_idx,
            activations=self.activations, n_candidates=self.selection.n_candidates,
            sorted_magnitudes=torch.from_numpy(self.sorted_magnitudes), pick=self.pick,
        )


def entropy_bits(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> Float[np.ndarray, "n_hits"]:
    return -(probabilities * np.log2(np.where(probabilities > 0, probabilities, 1.0))).sum(axis=1)


def confident_hits(probabilities: Float[np.ndarray, "n_hits n_clusters"], max_entropy: float) -> Bool[np.ndarray, "n_hits"]:
    return entropy_bits(probabilities) < max_entropy


def space_is_complete(space_dir: Path) -> bool:
    return all((space_dir / name).exists() for name in SPACE_FILES)


def save_hits(
    hadamards: Float[np.ndarray, "n_hits hidden"], selection: HitSelection, sample_ids: list[SampleId],
    noise: Float[np.ndarray, "n_noise"], space_dir: Path,
) -> None:
    np.save(space_dir / HADAMARDS, hadamards.astype(np.float32))
    np.savez(
        space_dir / HITS, batch_idx=selection.batch_idx, token_idx=selection.token_idx,
        activations=selection.activations, sample_ids=np.array(sample_ids),
    )
    arrays = pick_arrays(selection.pick)
    record = SelectionRecord(
        polarity=selection.polarity, n_candidates=selection.n_candidates,
        pick_fields=selection.pick.model_dump(exclude=set(arrays)),
    )
    (space_dir / SELECTION).write_text(record.model_dump_json())
    np.savez(space_dir / PICK_ARRAYS, **arrays)
    np.save(space_dir / SORTED_MAGNITUDES, selection.sorted_magnitudes.numpy())
    np.save(space_dir / NOISE, noise)


def save_distances(distances: Float[np.ndarray, "n_hits n_hits"], space_dir: Path) -> None:
    np.save(space_dir / DISTANCES, distances.astype(np.float32))


def save_forest(
    forest: DensityPeakForest, trees: ForestTrees, probabilities: Float[np.ndarray, "n_distinct n_clusters"],
    space_dir: Path,
) -> None:
    """`probabilities` are per distinct point, as `calculate_probs` returns them."""
    np.savez(space_dir / FOREST, **forest.model_dump())
    np.savez(
        space_dir / TREES,
        roots=trees.roots, tree=trees.tree, depth=trees.depth, cut_rho=trees.cuts.rho, cut_delta=trees.cuts.delta,
    )
    np.save(space_dir / PROBABILITIES, probabilities)


def load_space(space_dir: Path) -> HitSpace:
    hits, forest, trees = np.load(space_dir / HITS), np.load(space_dir / FOREST), np.load(space_dir / TREES)
    distinct_of_hit = forest["distinct_of_hit"]
    selection = SelectionRecord.model_validate_json((space_dir / SELECTION).read_text())
    arrays = dict(np.load(space_dir / PICK_ARRAYS))
    return HitSpace(
        hadamards=np.load(space_dir / HADAMARDS), batch_idx=hits["batch_idx"], token_idx=hits["token_idx"],
        activations=hits["activations"], sample_ids=hits["sample_ids"].tolist(),
        selection=selection, pick=load_pick(selection, arrays),
        sorted_magnitudes=np.load(space_dir / SORTED_MAGNITUDES), noise=np.load(space_dir / NOISE),
        probabilities=np.load(space_dir / PROBABILITIES)[distinct_of_hit], distinct_of_hit=distinct_of_hit,
        roots=trees["roots"],
    )


def load_density_graph(space_dir: Path) -> DensityGraph:
    forest = np.load(space_dir / FOREST)
    return DensityGraph(
        kth_distance=forest["kth_distance"], counts=forest["counts"], neighbours=forest["neighbours"],
        distinct_of_hit=forest["distinct_of_hit"],
    )


def load_distances(space_dir: Path) -> Float[np.ndarray, "n_hits n_hits"]:
    return np.load(space_dir / DISTANCES, mmap_mode="r")
