#!/usr/bin/env python
"""How much of a big run's clustered hits a smaller run's HDBSCAN still clusters.

- The small run's rows are rebuilt with its config, and cuml refits them with `prediction_data`.
  - Its labels are checked against the small report's cluster sizes.
- The big run's rows and labels come from its `hits_cache/` (see `the_logit_cluster_summary.py`).
- `approximate_predict` places every big-run hit in a small-run cluster, or in noise.

    python scripts/hdbscan_predict_coverage.py --config <yaml> --hits.max-hits 250000 \\
        --out-dir <small run dir> --big-run <big run dir>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from jaxtyping import Int

sys.path.insert(0, str(Path(__file__).parent))
from hdbscan_soft_compare import report_rows  # noqa: E402

from in_tfm.neuron_run import report_name  # noqa: E402
from in_tfm.run_config import HdbscanConfig, load_run_config  # noqa: E402

WATCH = [519, 304, 647, 537, 525]


def check_sizes(labels: Int[np.ndarray, "n"], report_dir: Path) -> None:
    meta = json.loads((report_dir / "meta.json").read_text())
    expected = {int(cid): c["n_hits"] for cid, c in meta["clusters"].items()}
    found = Counter(labels[labels >= 0].tolist())
    mismatched = [cid for cid, n in expected.items() if found.get(cid) != n]
    print(
        f"refit: {len(expected)} report clusters, {len(mismatched)} with a different size"
    )


def purity(
    big: Int[np.ndarray, "n"], predicted: Int[np.ndarray, "n"]
) -> dict[int, tuple[float, float, int]]:
    """Per big cluster: share placed in any small cluster, share in its most common one, and that cluster."""
    rows = {}
    for cid in np.unique(big[big >= 0]):
        landed = predicted[big == cid]
        clustered = landed[landed >= 0]
        if len(clustered) == 0:
            rows[int(cid)] = (0.0, 0.0, -1)
            continue
        top, n = Counter(clustered.tolist()).most_common(1)[0]
        rows[int(cid)] = (len(clustered) / len(landed), n / len(landed), int(top))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--big-run", type=Path, required=True)
    known, rest = parser.parse_known_args()
    config = load_run_config(rest)
    strategy = config.clustering.strategy
    assert isinstance(strategy, HdbscanConfig)

    from cuml.cluster import HDBSCAN
    from cuml.cluster.hdbscan import approximate_predict

    small_rows = report_rows(config)
    clusterer = HDBSCAN(
        min_cluster_size=strategy.min_cluster_size,
        min_samples=strategy.min_samples,
        cluster_selection_method=strategy.selection_method,
        prediction_data=True,
    ).fit(small_rows)
    small_labels = np.asarray(clusterer.labels_)
    [neuron_idx] = config.neuron_idxs
    check_sizes(
        small_labels, Path(config.out_dir) / report_name(neuron_idx, config.polarity)
    )

    cache = known.big_run / "hits_cache"
    big_rows = np.load(cache / "rows.npy")
    big_labels = np.load(cache / "labels.npy")
    predicted, _ = approximate_predict(clusterer, big_rows)
    predicted = np.asarray(predicted)

    clustered = big_labels >= 0
    caught = predicted >= 0
    print(f"big run: {int(clustered.sum()):,} clustered hits of {len(big_labels):,}")
    print(
        f"  clustered in big, placed in a small cluster: {caught[clustered].mean():.1%}"
    )
    print(
        f"  clustered in big, small calls noise:        {(~caught[clustered]).mean():.1%}"
    )
    print(
        f"  noise in big, placed in a small cluster:    {caught[~clustered].mean():.1%}"
    )

    per_cluster = purity(big_labels, predicted)
    sizes = Counter(big_labels[clustered].tolist())
    coverage = np.array([per_cluster[c][0] for c in sizes])
    weights = np.array([sizes[c] for c in sizes])
    for threshold in [0.9, 0.5, 0.1]:
        kept = coverage >= threshold
        print(
            f"  big clusters with >= {threshold:.0%} of hits placed: {kept.mean():.1%} of clusters, "
            f"{weights[kept].sum() / weights.sum():.1%} of clustered hits"
        )
    for cid in WATCH:
        placed, top_share, top = per_cluster[cid]
        print(
            f"  c{cid} n={sizes[cid]}: {placed:.1%} placed, {top_share:.1%} in small c{top}"
        )


if __name__ == "__main__":
    main()
