#!/usr/bin/env python
"""Collects one neuron's hits once, and saves what the cluster-space analysis iterates on.

    python scripts/collect_hit_space.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --inception.image-dir /content/data/inception_pool --out-dir /content/runs/mixed4e_55

Everything lands in `<out_dir>/space/`, so later analyses load arrays and never touch the model:
- `hadamards.npy` and `hits.npz`: each hit's Hadamard row, and where it came from (image, grid position, activation).
- `distances.npy`: the full hit-by-hit distance matrix, float32.
- `forest.npz`, `trees.npz` and `probabilities.npy`: the density-peak forest, its trees, and each distinct point's probability per tree.
"""

import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from run_neuron_report import load_adapter, load_samples

from in_tfm.density_peaks import calculate_probs
from in_tfm.device import empty_cache
from in_tfm.hit_selection import HitSelection, select_hits
from in_tfm.neuron_capture import merge_positions
from in_tfm.neuron_run import clustered_rows, density_peak_trees, layer_getter_for, prepare_run, timed
from in_tfm.run_config import DensityPeaksConfig, RunConfig, load_run_config
from in_tfm.weighted_jaccard.vectorized import weighted_jaccard_distances


class CollectedHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hadamards: Float[np.ndarray, "n_hits hidden"]
    selection: HitSelection
    sample_ids: list[str]


def collect_hits(config: RunConfig, samples: list[str], neuron_idx: int) -> CollectedHits:
    run = prepare_run(config, load_adapter(config), samples)
    with timed("pass 1: scan"):
        scan = run.capture.scan()
    selection = select_hits(scan, neuron_idx, "positive", config.hits, config.seed)
    merged = merge_positions([(selection.batch_idx, selection.token_idx)], scan.valid_mask.shape[1])
    with timed("pass 2: gather"):
        gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    run.model.to("cpu")
    empty_cache()
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)
    rows = gathered.inputs[merged.index_maps[0]]
    hadamards = clustered_rows(rows, weight, neuron_idx, config.clustering)
    return CollectedHits(hadamards=hadamards, selection=selection, sample_ids=scan.sample_ids)


def save_hits(hits: CollectedHits, space_dir: Path) -> None:
    np.save(space_dir / "hadamards.npy", hits.hadamards.astype(np.float32))
    selection = hits.selection
    np.savez(
        space_dir / "hits.npz",
        batch_idx=selection.batch_idx, token_idx=selection.token_idx, activations=selection.activations,
        sample_ids=np.array(hits.sample_ids),
    )


def save_distances(hadamards: Float[np.ndarray, "n_hits hidden"], space_dir: Path) -> None:
    distances = weighted_jaccard_distances(hadamards.astype(np.float64), normalise=False)
    np.save(space_dir / "distances.npy", distances.astype(np.float32))


def save_forest(hadamards: Float[np.ndarray, "n_hits hidden"], config: RunConfig, space_dir: Path) -> None:
    strategy = config.clustering.strategy
    assert isinstance(strategy, DensityPeaksConfig), "the cluster space is read from the density-peak forest"
    forest, trees = density_peak_trees(hadamards, config.clustering, strategy)
    np.savez(space_dir / "forest.npz", **forest.model_dump())
    np.savez(
        space_dir / "trees.npz",
        roots=trees.roots, tree=trees.tree, depth=trees.depth, cut_rho=trees.cuts.rho, cut_delta=trees.cuts.delta,
    )
    np.save(space_dir / "probabilities.npy", calculate_probs(forest, trees, strategy.peak_share))


def main() -> None:
    config = load_run_config(sys.argv[1:])
    neuron_idx = config.neuron_idxs[0]
    space_dir = Path(config.out_dir) / "space"
    space_dir.mkdir(parents=True, exist_ok=True)

    with timed("collect hits"):
        hits = collect_hits(config, load_samples(config), neuron_idx)
    print(f"{len(hits.hadamards)} hits, hadamards {hits.hadamards.shape}")
    save_hits(hits, space_dir)
    with timed("distance matrix"):
        save_distances(hits.hadamards, space_dir)
    with timed("forest and probabilities"):
        save_forest(hits.hadamards, config, space_dir)
    print(f"saved {sorted(p.name for p in space_dir.iterdir())} -> {space_dir}")


if __name__ == "__main__":
    main()
