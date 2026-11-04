#!/usr/bin/env python
"""Hard against soft HDBSCAN on the hits of one neuron-report config.

- The hits and rows are rebuilt with the report's own selection and seed, so they are the report's.
- cuml fits once with `prediction_data`, which gives the hard labels and the membership vectors together.
- A point is soft noise when its largest membership is below a threshold, for several thresholds.
- Entropy is over the membership vector rescaled to sum to 1, in bits.

    python scripts/hdbscan_soft_compare.py --config configs/tiny_attn_only_1l_the_logit.yaml
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from run_neuron_report import load_adapter, load_samples  # noqa: E402

from in_tfm.hit_selection import select_hits  # noqa: E402
from in_tfm.neuron_run import clustered_rows, layer_getter_for, prepare_run  # noqa: E402
from in_tfm.run_config import HdbscanConfig, RunConfig, load_run_config  # noqa: E402

SOFT_NOISE_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]


class SoftFit(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    hard: Int[np.ndarray, "n"]
    membership: Float[np.ndarray, "n k"]


def report_rows(config: RunConfig) -> Float[np.ndarray, "n_hits hidden"]:
    [neuron_idx] = config.neuron_idxs
    run = prepare_run(config, load_adapter(config), load_samples(config))
    scan = run.capture.scan()
    selection = select_hits(scan, neuron_idx, config.polarity, config.hits, config.seed)
    gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
    layer_weight = layer_getter_for(config)(run.module).weight
    weight = run.layout.weight(layer_weight)
    return clustered_rows(gathered.inputs, weight, neuron_idx, config.clustering)


def soft_fit(rows: Float[np.ndarray, "n d"], strategy: HdbscanConfig) -> SoftFit:
    from cuml.cluster import HDBSCAN
    from cuml.cluster.hdbscan import all_points_membership_vectors

    clusterer = HDBSCAN(
        min_cluster_size=strategy.min_cluster_size,
        min_samples=strategy.min_samples,
        cluster_selection_method=strategy.selection_method,
        prediction_data=True,
    ).fit(rows)
    return SoftFit(
        hard=np.asarray(clusterer.labels_),
        membership=np.asarray(all_points_membership_vectors(clusterer)),
    )


def entropy_bits(membership: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n"]:
    p = membership / membership.sum(1, keepdims=True).clip(min=1e-12)
    return -(p * np.log2(p.clip(min=1e-12))).sum(1)


def quantiles(values: Float[np.ndarray, "n"]) -> dict[str, float]:
    if len(values) == 0:
        return {}
    qs = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "mean": round(float(values.mean()), 4),
        **{
            f"q{int(q * 100)}": round(float(v), 4)
            for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], qs)
        },
    }


def summary(fit: SoftFit) -> dict:
    n, k = fit.membership.shape
    top = fit.membership.max(1)
    entropy = entropy_bits(fit.membership)
    hard_noise = fit.hard < 0
    rows: dict = {
        "n_points": n,
        "n_clusters": k,
        "max_entropy_bits": round(float(np.log2(k)), 3),
        "hard_noise": int(hard_noise.sum()),
        "row_sum": quantiles(fit.membership.sum(1)),
        "top_membership": quantiles(top),
        "argmax_agrees_with_hard": round(
            float(
                (fit.membership[~hard_noise].argmax(1) == fit.hard[~hard_noise]).mean()
            ),
            4,
        ),
        "by_threshold": [],
    }
    for threshold in SOFT_NOISE_THRESHOLDS:
        assigned = top >= threshold
        rows["by_threshold"].append(
            {
                "threshold": threshold,
                "soft_noise": int((~assigned).sum()),
                "hard_noise_now_assigned": int((assigned & hard_noise).sum()),
                "hard_clustered_now_noise": int((~assigned & ~hard_noise).sum()),
                "entropy_assigned": quantiles(entropy[assigned]),
                "entropy_assigned_hard_clustered": quantiles(
                    entropy[assigned & ~hard_noise]
                ),
                "entropy_assigned_hard_noise": quantiles(
                    entropy[assigned & hard_noise]
                ),
            }
        )
    return rows


def main() -> None:
    config = load_run_config(sys.argv[1:])
    assert isinstance(config.clustering.strategy, HdbscanConfig)
    started = time.time()
    rows = report_rows(config)
    print(f"[rows {rows.shape}] {time.time() - started:.1f}s", flush=True)
    fit = soft_fit(rows, config.clustering.strategy)
    print(f"[soft fit {fit.membership.shape}] {time.time() - started:.1f}s", flush=True)
    result = summary(fit)
    out = Path(config.out_dir) / "hdbscan_soft"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "soft.npz",
        hard=fit.hard,
        top=fit.membership.max(1),
        argmax=fit.membership.argmax(1),
        row_sum=fit.membership.sum(1),
        entropy=entropy_bits(fit.membership),
    )
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"[done, total] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
