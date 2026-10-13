#!/usr/bin/env python
"""Clusters of every neuron of one layer type of one grokking state, with no HTML reports.

    uv run python scripts/cluster_grokking_neurons.py --state 40000 --target up_proj --out experiments_grokking/final_up_proj.npz

- The samples are all 113 x 113 pairs, so a cluster's hits are positions of `a b =` that can be read as a set of pairs.
- Each neuron stores its hit positions, activations and cluster labels (-1 is noise), which is all the circuit analysis reads.
"""

from pathlib import Path

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.clustering import cluster_labels
from in_tfm.device import empty_cache
from in_tfm.hadamard import hadamard_from_rows
from in_tfm.hit_selection import select_hits
from in_tfm.models.grokking import GrokkingAdapter
from in_tfm.models.grokking.data import pairs_for
from in_tfm.neuron_capture import merge_positions
from in_tfm.neuron_run import layer_getter_for, polarities_for, prepare_run, timed
from in_tfm.neuron_tracking import state_path
from in_tfm.run_config import load_run_config


class ClusterNeuronsConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    state: int
    target: str = "up_proj"
    n_neurons: int = Field(512, description="neurons 0..n_neurons-1")
    chunk: int = 32
    pairs: str = "all"
    states_dir: Path = Path("checkpoints/grokking/states")
    out: Path


def cluster_chunk(state: int, target: str, pairs: str, neurons: list[int], states_dir: Path) -> dict[str, np.ndarray]:
    argv = ["--config", "configs/grokking_v_proj.yaml", "--target", target, "--grok.pairs", pairs, "--hits.max-hits", "10000"]
    argv += ["--grok.checkpoint", str(state_path(states_dir, state))]
    for n in neurons:
        argv += ["--neurons", str(n)]
    config = load_run_config(argv)
    adapter = GrokkingAdapter.from_pretrained(str(config.grok.checkpoint))
    run = prepare_run(config, adapter, pairs_for(config.grok.pairs, config.grok.train_fraction, config.grok.data_seed))
    scan = run.capture.scan()
    selections = {(n, p): select_hits(scan, n, p, config.hits, config.seed) for n in neurons for p in polarities_for(config)}
    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], scan.valid_mask.shape[1])
    gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    run.model.to("cpu")
    empty_cache()
    weight = layer_getter_for(config)(run.module).weight
    out: dict[str, np.ndarray] = {}
    for (neuron, polarity), index_map in zip(selections, merged.index_maps):
        selection = selections[(neuron, polarity)]
        rows = gathered.inputs[index_map]
        clustering = config.clustering
        labels = cluster_labels(hadamard_from_rows(rows, weight, neuron), clustering.min_cluster_size, clustering.selection_method)
        key = f"{neuron}_{polarity}"
        out[f"{key}_batch"] = selection.batch_idx
        out[f"{key}_token"] = selection.token_idx
        out[f"{key}_label"] = np.asarray(labels)
        out[f"{key}_act"] = selection.activations
    return out


def main() -> None:
    config = ClusterNeuronsConfig()
    results: dict[str, np.ndarray] = {}
    for start in range(0, config.n_neurons, config.chunk):
        neurons = list(range(start, min(start + config.chunk, config.n_neurons)))
        with timed(f"neurons {neurons[0]}..{neurons[-1]}"):
            results |= cluster_chunk(config.state, config.target, config.pairs, neurons, config.states_dir)
        config.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(config.out, **results)
    print("DONE", config.out)


if __name__ == "__main__":
    main()
