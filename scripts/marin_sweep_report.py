#!/usr/bin/env python
"""One Marin 32B neuron report, after an HDBSCAN sweep over its gathered hits.

- The forward passes run once. Every HDBSCAN setting then runs on the same Hadamards.
- The report uses the setting with the lowest noise share, among those that pass 2 limits:
  - at most `MAX_CLUSTERS` clusters, since more would make AttnLRP over every cluster too slow on an L4
  - a largest cluster of at most `MAX_LARGEST_SHARE` of the hits, since low noise alone rewards one giant cluster
- The sweep is saved beside the report, as `sweep.json`.

    python scripts/marin_sweep_report.py --config configs/marin_l16_19311.yaml
    python scripts/marin_sweep_report.py --config configs/marin_l16_19311.yaml \\
        --layer-name layers.16.mlp.up_proj --out-dir reports/marin-32b/layers.16.mlp.up_proj
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel

from in_tfm.clustering import cluster_labels
from in_tfm.models.qwen3 import Qwen3Adapter
from in_tfm.neuron_run import (
    clustered_rows,
    collect_hits,
    prepare_run,
    report_collected,
    report_name,
    timed,
)
from in_tfm.run_config import HdbscanConfig, RunConfig, load_run_config
from in_tfm.sources.wikitext import load_texts

GRID = [(12, 12), (25, 25), (25, 8), (50, 50), (50, 15), (100, 25)]
"""(min_cluster_size, min_samples) pairs."""
MAX_CLUSTERS = 1500
MAX_LARGEST_SHARE = 0.25


class SweepRow(BaseModel):
    min_cluster_size: int
    min_samples: int
    n_clusters: int
    noise_share: float
    largest_share: float
    seconds: float


def sweep_row(
    hdmd: Float[np.ndarray, "n_hits hidden"], min_cluster_size: int, min_samples: int
) -> SweepRow:
    start = time.perf_counter()
    labels = cluster_labels(hdmd, min_cluster_size, "leaf", min_samples)
    clustered = labels[labels >= 0]
    return SweepRow(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_clusters=len(np.unique(clustered)),
        noise_share=float((labels < 0).mean()),
        largest_share=float(np.bincount(clustered).max() / len(labels))
        if len(clustered)
        else 0.0,
        seconds=time.perf_counter() - start,
    )


def chosen(rows: list[SweepRow]) -> SweepRow:
    """Ties go to the larger clusters. If no setting passes, the one with the smallest largest cluster."""
    allowed = [
        r
        for r in rows
        if r.n_clusters <= MAX_CLUSTERS and r.largest_share <= MAX_LARGEST_SHARE
    ]
    if not allowed:
        return min(rows, key=lambda r: r.largest_share)
    return min(allowed, key=lambda r: (r.noise_share, -r.min_cluster_size))


def with_hdbscan(config: RunConfig, row: SweepRow) -> RunConfig:
    strategy = HdbscanConfig(
        min_cluster_size=row.min_cluster_size,
        min_samples=row.min_samples,
        selection_method="leaf",
    )
    clustering = config.clustering.model_copy(update={"strategy": strategy})
    return config.model_copy(update={"clustering": clustering})


def main() -> None:
    start = time.perf_counter()
    config = load_run_config(sys.argv[1:])
    if len(config.neuron_idxs) != 1:
        raise ValueError("the sweep picks one setting per run, so give one neuron")

    with timed("load samples"):
        texts = load_texts(config.text, config.seed)
    with timed("load model"):
        adapter = Qwen3Adapter.from_pretrained(
            config.model.value,
            config.layer_name,
            config.text.max_length,
            device=config.device,
        )
    run = prepare_run(config, adapter, texts)
    collected = collect_hits(config, run)

    [((neuron_idx, polarity), rows)] = collected.rows.items()
    hdmd = clustered_rows(rows, collected.weight, neuron_idx, config.clustering)
    sweep = []
    for min_cluster_size, min_samples in GRID:
        row = sweep_row(hdmd, min_cluster_size, min_samples)
        print(f"[sweep] {row.model_dump_json()}", flush=True)
        sweep.append(row)
    del hdmd
    best = chosen(sweep)
    print(f"[sweep] chosen {best.model_dump_json()}", flush=True)

    report_collected(with_hdbscan(config, best), run, collected)
    out = Path(config.out_dir) / report_name(neuron_idx, polarity)
    (out / "sweep.json").write_text(
        json.dumps(
            {"chosen": best.model_dump(), "rows": [r.model_dump() for r in sweep]},
            indent=2,
        )
    )
    print(f"[total] {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
