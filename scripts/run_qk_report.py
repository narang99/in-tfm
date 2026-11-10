#!/usr/bin/env python
"""Clusters one layer's attention scores, split across head dimensions, one page per head.

Settings are in `in_tfm.run_config.QkRunConfig`, with defaults from `--config <yaml>`:

    python scripts/run_qk_report.py --config configs/qwen3_32b_l55_qk.yaml --heads 3
"""

import sys
import time

import torch

from in_tfm.cluster_membership import most_probable_labels
from in_tfm.models.qwen3 import Qwen3Adapter
from in_tfm.neuron_run import cluster_hits, forest_device, timed
from in_tfm.qk_pairs import (
    elbow_pairs,
    gather_rows,
    scan_pairs,
    worst_gather_mismatch,
)
from in_tfm.qk_report import PairReportInputs, write_pair_report
from in_tfm.run_config import load_qk_run_config
from in_tfm.sources.wikitext import load_texts


def main() -> None:
    start = time.perf_counter()
    config = load_qk_run_config(sys.argv[1:])
    with timed("load samples"):
        texts = load_texts(config.text, config.seed)
    with timed("load model"):
        adapter = Qwen3Adapter.from_pretrained(
            config.model.value,
            config.layer_name,
            config.text.max_length,
            device=config.device,
        )
    model = adapter.get_model()
    source = adapter.make_source(texts)

    with timed("pass 1: scan"):
        scan = scan_pairs(
            model, source, config.layer, config.heads, config.batch_size, config.device
        )
    hits = []
    for head in config.heads:
        # Sorted on the CPU: a head has hundreds of millions of positive scores,
        # and the model still fills the GPU until the gather is done.
        head_hits = elbow_pairs(scan, head, config.max_hits, config.seed, "cpu")
        print(
            f"[head {head}] elbow cut-off {head_hits.cutoff:.4f}, "
            f"{head_hits.n_candidates} pairs above it, {len(head_hits.sample_idx)} kept"
        )
        hits.append(head_hits)
    with timed("pass 2: gather"):
        gathered = gather_rows(
            model, source, config.layer, scan, hits, config.batch_size, config.device
        )
    model.to(torch.device("cpu"))
    token_ids = source.token_ids(scan.sample_ids)

    for pair_rows in gathered:
        head = pair_rows.hits.head
        print(
            f"[head {head}] row sums match the scan's scores to {worst_gather_mismatch(pair_rows):.2e}"
        )
        with timed(f"head {head}: cluster"):
            clustered = cluster_hits(
                pair_rows.rows.numpy(), config.clustering, forest_device(config.device)
            )
        labels = most_probable_labels(clustered.memberships)
        print(f"[head {head}] {labels.max() + 1} clusters, {(labels < 0).sum()} noise")
        path = write_pair_report(
            config.out_dir / f"head_{head}",
            f"{config.model.value} layer {config.layer} head {head}: q ⊙ k pairs",
            PairReportInputs(
                pair_rows=pair_rows,
                labels=labels,
                sample_ids=[str(i) for i in scan.sample_ids],
                token_ids=token_ids,
            ),
            adapter.tokenizer.decode,
            adapter.tokenizer.pad_token_id,
            config.report.examples_per_cluster,
            config.report.context_tokens,
            config.seed,
        )
        print(f"[head {head}] report -> {path}")
    print(f"total {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
