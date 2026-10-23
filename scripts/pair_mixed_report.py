#!/usr/bin/env python
"""The pair-by-pair mixed membership report, from the arrays `collect_hit_space.py` saved.

    python scripts/pair_mixed_report.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --inception.image-dir /content/data/inception_pool --out-dir /content/runs/mixed4e_55

It reads `<out_dir>/space/` and writes `<out_dir>/pair_report/`: an `index.html` and a page per pair.
Only the model pass for the cards runs here, so clustering is not repeated.
"""

import sys
from pathlib import Path

import numpy as np
from run_neuron_report import load_adapter, load_samples

from in_tfm.hadamard import near_square_shape
from in_tfm.hit_selection import HitSelection
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.neuron_run import PreparedRun, layer_getter_for, max_entropy_of, prepare_run, timed
from in_tfm.pair_mixed_report import write_pair_mixed_pages
from in_tfm.run_config import RunConfig, load_run_config


def saved_hits(config: RunConfig, run: PreparedRun, space: Path, probabilities: np.ndarray) -> NeuronClusterHits:
    arrays = np.load(space / "hits.npz")
    hadamards = np.load(space / "hadamards.npy")
    # only the hit positions are read from the selection, so the elbow curve and pick are left out
    selection = HitSelection.model_construct(
        polarity="positive", batch_idx=arrays["batch_idx"], token_idx=arrays["token_idx"], activations=arrays["activations"],
    )
    return NeuronClusterHits(
        model=run.module, source=run.source, presenter=run.presenter, sample_ids=arrays["sample_ids"].tolist(),
        layer_getter=layer_getter_for(config), neuron_idx=config.neuron_idxs[0], selection=selection,
        labels=probabilities.argmax(axis=1), hdmds=hadamards, hadamard_shape=near_square_shape(hadamards.shape[1]),
        device=config.device,
    )


def main() -> None:
    config = load_run_config(sys.argv[1:])
    space = Path(config.out_dir) / "space"
    forest = np.load(space / "forest.npz")
    probabilities = np.load(space / "probabilities.npy")[forest["distinct_of_hit"]]
    entropy = -(probabilities * np.log2(np.where(probabilities > 0, probabilities, 1.0))).sum(axis=1)
    confident = entropy < max_entropy_of(config.clustering)

    run = prepare_run(config, load_adapter(config), load_samples(config))
    hits = saved_hits(config, run, space, probabilities)
    with timed("pair pages"):
        index = write_pair_mixed_pages(
            hits, Path(config.out_dir) / "pair_report", probabilities, entropy, confident, run.attr_fn,
        )
    print(f"pair index -> {index}")


if __name__ == "__main__":
    main()
