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
    PairScan,
    elbow_pairs,
    gather_rows,
    scan_pairs,
    worst_gather_mismatch,
)
from in_tfm.qk_report import PairReportInputs, write_pair_report
from in_tfm.run_config import QkRunConfig, load_qk_run_config
from in_tfm.sources.wikitext import load_texts


def scanned_or_resumed(config: QkRunConfig, model, source) -> PairScan:
    """Pass 1 is saved as soon as it ends, so a crash later costs only what came after it.

    - A saved scan is reused only for the same heads and sample count.
    """
    path = config.out_dir / "scan.pt"
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved.heads == config.heads and len(saved.sample_ids) == len(source.texts):
            print(f"resumed pass 1 from {path}")
            return saved
        print(f"{path} is for other heads or samples, scanning again")
    with timed("pass 1: scan"):
        scan = scan_pairs(
            model, source, config.layer, config.heads, config.batch_size, config.device
        )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    with timed(f"save scan -> {path}"):
        torch.save(scan, path)
    return scan


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

    scan = scanned_or_resumed(config, model, source)
    hits = []
    for head in config.heads:
        head_hits = elbow_pairs(
            scan, head, config.max_hits, config.seed, config.exclude_first_key
        )
        print(
            f"[head {head}] elbow cut-off {head_hits.cutoff:.4f}, "
            f"{head_hits.n_candidates} pairs above it, {len(head_hits.sample_idx)} kept"
        )
        hits.append(head_hits)
    with timed("pass 2: gather"):
        gathered = gather_rows(model, source, config.layer, scan, hits, config.device)
    model.to(torch.device("cpu"))
    token_ids = source.token_ids(scan.sample_ids)

    for pair_rows in gathered:
        head = pair_rows.hits.head
        print(
            f"[head {head}] row sums match the scan's float16 scores to a relative {worst_gather_mismatch(pair_rows):.2e}"
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
