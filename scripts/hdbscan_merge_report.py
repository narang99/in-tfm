#!/usr/bin/env python
"""One page showing how HDBSCAN merges a writer neuron's hits, with a cluster card at every node.

The hits are the ones `writer_neuron_report.py` picks with the stream-writers walk. Its saved picks are reused.
HDBSCAN is fit once on the config's clustering distance, so the leaf and eom clusters come from the same tree.

    python scripts/hdbscan_merge_report.py --config configs/tinystories_1m_l6_mlp_write.yaml \\
        --picks-from reports/tinystories-1m/h.6.mlp.c_proj --min-cluster-size 10 \\
        --out reports/tinystories-1m/h.6.mlp.c_proj_hdbscan_merges
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from writer_neuron_report import (
    parse_walk_args,
    picked_columns,
    saved_or_walked,
    tinystories,
    walk_selection,
)

from in_tfm.activation_ranges import hit_activations
from in_tfm.hadamard import near_square_shape
from in_tfm.hdbscan_merges import (
    fit_tree,
    leaf_and_eom_only,
    merge_nodes,
    shown_hits,
)
from in_tfm.html_report import page
from in_tfm.layers import submodule_getter
from in_tfm.merge_page import TRIMMED_NOTE, merge_page_body, node_names
from in_tfm.models.gpt_neo import GptNeoAdapter
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.neuron_run import clustered_rows, prepare_run, timed
from in_tfm.presenters import ClusterHit
from in_tfm.presenters.artifacts import ClusterArtifacts
from in_tfm.run_config import load_run_config
from in_tfm.weighted_jaccard import weighted_jaccard_distances

PER_CARD = 6


def parse_merge_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--picks-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument(
        "--leaf-and-eom-only",
        action="store_true",
        help="draw only the leaf and eom clusters, each leaf straight under its eom cluster",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    merge_args, rest = parse_merge_args(sys.argv[1:])
    walk_args, rest = parse_walk_args(rest)
    config = load_run_config(rest)
    neuron_idx = config.neuron_idxs[0]
    stories = tinystories(walk_args.n_stories)
    adapter = GptNeoAdapter.from_pretrained(config.model.value, config.text.max_length)
    picks = saved_or_walked(
        config.model_copy(update={"out_dir": merge_args.picks_from}),
        adapter.tokenizer,
        stories,
        walk_args.share,
        walk_args.workers,
    )

    run = prepare_run(config, adapter, stories)
    scan = run.capture.scan()
    picked = picked_columns(
        picks, 0, scan.valid_mask, adapter.tokenizer.padding_side == "right"
    )
    selection = walk_selection(scan, neuron_idx, picked, config)
    gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
    layer_weight = submodule_getter(config.layer_name)(run.module).weight
    assert isinstance(layer_weight, torch.Tensor)
    hdmd = clustered_rows(
        gathered.inputs, run.layout.weight(layer_weight), neuron_idx, config.clustering
    )

    with timed("distances and tree"):
        distances = weighted_jaccard_distances(
            hdmd, normalise=config.clustering.normalise
        )
        clusterer = fit_tree(
            distances, merge_args.min_cluster_size, merge_args.min_samples
        )
        del distances
    nodes = merge_nodes(clusterer)
    drawn = leaf_and_eom_only(nodes) if merge_args.leaf_and_eom_only else nodes
    names = node_names(drawn)
    print(
        f"{len(nodes)} nodes, {sum(n.is_leaf for n in nodes.values())} leaves, "
        f"{sum(n.is_eom for n in nodes.values())} eom clusters"
    )

    hits = NeuronClusterHits(
        model=run.module,
        source=run.source,
        presenter=run.presenter,
        sample_ids=scan.sample_ids,
        layer_getter=submodule_getter(config.layer_name),
        neuron_idx=neuron_idx,
        selection=selection,
        labels=clusterer.labels_,
        hdmds=hdmd,
        hadamard_shape=near_square_shape(hdmd.shape[1]),
        device=config.device,
    )
    attributed: dict[int, ClusterHit] = {}
    _, copy_of = np.unique(hdmd.round(6), axis=0, return_inverse=True)

    def card(node: int) -> str:
        shown = shown_hits(nodes, node, PER_CARD, copy_of.ravel())
        for i in shown:
            if i not in attributed:
                attributed[i] = hits.cluster_hit(i, run.attr_fn)
        artifacts = ClusterArtifacts(cluster_dir=merge_args.out / names[node])
        artifacts.cluster_dir.mkdir(parents=True, exist_ok=True)
        return run.presenter.render(
            [attributed[i] for i in shown], artifacts, hits.hadamard_shape
        )

    sample_ids = [scan.sample_ids[b] for b in selection.batch_idx]
    tokens = [
        s[1]
        for s in run.source.sentences_around(sample_ids, selection.token_idx.tolist())
    ]
    activations = hit_activations(
        scan, neuron_idx, selection.batch_idx, selection.token_idx
    )
    heading = (
        f"Neuron {neuron_idx}: HDBSCAN {'leaf and eom clusters' if merge_args.leaf_and_eom_only else 'merges'}, "
        f"min_cluster_size {merge_args.min_cluster_size}, "
        f"{config.clustering.distance}"
    )
    with timed("cards"):
        body = merge_page_body(
            drawn,
            card,
            tokens,
            np.asarray(activations),
            heading,
            run.presenter.page_controls(),
            TRIMMED_NOTE if merge_args.leaf_and_eom_only else "",
        )
    path = merge_args.out / "index.html"
    path.write_text(page(heading, body))
    print(f"merges -> {path}")


if __name__ == "__main__":
    main()
