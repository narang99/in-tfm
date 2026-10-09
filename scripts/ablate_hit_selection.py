#!/usr/bin/env python
"""Elbow vs bucketed hit selection, on the same scan, with the same budget.

- Takes the flags of `run_neuron_report.py` (`--hits.method` is ignored, both are run).
- Skips attribution and rendering: only selection, gather and clustering matter here.
- Writes `ablation.json` and `ablation.md` to `--out-dir`.
"""

import sys
import time

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from scipy.optimize import linear_sum_assignment
from transformers import PreTrainedTokenizerBase

import run_neuron_report as report
from in_tfm.bucketing import merged_bucket_ids
from in_tfm.clustering import cluster_labels
from in_tfm.hadamard import hadamard_from_rows
from in_tfm.hit_selection import HitSelection, select_hits
from in_tfm.neuron_capture import ScanResult, merge_positions
from in_tfm.neuron_run import layer_getter_for, prepare_run
from in_tfm.run_config import HitSelectionConfig, load_run_config
from in_tfm.text_corpus import load_texts

MODES: dict[str, dict] = {
    "elbow": {"method": "elbow"},
    "bucketed": {"method": "bucketed"},
    "bucketed_floor10": {"method": "bucketed", "bucket_floor_fraction": 0.1},
}


class ModeResult(BaseModel):
    neuron: int
    mode: str
    threshold: float
    n_above_threshold: int
    n_kept: int
    cluster_seconds: float
    n_clusters: int
    noise_fraction: float
    median_activation: float
    max_activation: float
    share_above_elbow: float
    n_unique_tokens: int
    top_token: str
    top_token_share: float
    largest_cluster_share: float


def kept_activations(scan: ScanResult, neuron_idx: int, selection: HitSelection) -> np.ndarray:
    column = scan.neuron_column(neuron_idx)[..., 0].numpy()
    return column[selection.batch_idx, selection.token_idx]


def kept_token_ids(token_ids: torch.Tensor, selection: HitSelection) -> np.ndarray:
    return token_ids[selection.batch_idx, selection.token_idx].numpy()


def top_token_and_share(ids: np.ndarray, tokenizer: PreTrainedTokenizerBase) -> tuple[str, float]:
    values, counts = np.unique(ids, return_counts=True)
    top = counts.argmax()
    return repr(tokenizer.decode([int(values[top])])), float(counts[top] / len(ids))


def summarise(
    neuron_idx: int,
    mode: str,
    selection: HitSelection,
    activations: np.ndarray,
    token_ids: np.ndarray,
    labels: np.ndarray,
    cluster_seconds: float,
    elbow_value: float,
    tokenizer: PreTrainedTokenizerBase,
) -> ModeResult:
    real = labels[labels >= 0]
    sizes = np.bincount(real) if len(real) else np.array([0])
    top_token, top_share = top_token_and_share(token_ids, tokenizer)
    return ModeResult(
        neuron=neuron_idx,
        mode=mode,
        threshold=selection.threshold,
        n_above_threshold=selection.n_above_threshold,
        n_kept=len(labels),
        cluster_seconds=cluster_seconds,
        n_clusters=int((sizes > 0).sum()),
        noise_fraction=float((labels < 0).mean()),
        median_activation=float(np.median(activations)),
        max_activation=float(activations.max()),
        share_above_elbow=float((activations > elbow_value).mean()),
        n_unique_tokens=len(np.unique(token_ids)),
        top_token=top_token,
        top_token_share=top_share,
        largest_cluster_share=float(sizes.max() / len(labels)),
    )


class BucketRow(BaseModel):
    neuron: int
    mode: str
    bucket: int
    range_low: float
    range_high: float
    n_in_corpus: int
    n_kept: int
    n_noise: int
    clusters_touching: int
    clusters_homed: int


def bucket_breakdown(
    neuron_idx: int,
    mode: str,
    selection: HitSelection,
    corpus_values: np.ndarray,
    activations: np.ndarray,
    labels: np.ndarray,
    hits: HitSelectionConfig,
) -> tuple[list[BucketRow], str]:
    """`clusters_homed` counts clusters whose most common bucket is this one, so each cluster
    is counted once. `clusters_touching` counts every cluster with at least one hit here."""
    assert selection.bucket_ids is not None
    corpus = corpus_values[corpus_values > selection.threshold]
    corpus_ids = merged_bucket_ids(corpus, hits.n_buckets, selection.threshold, hits.min_bucket_size)
    kept_bucket = selection.bucket_ids
    n_groups = corpus_ids.max() + 1
    home = {c: np.bincount(kept_bucket[labels == c], minlength=n_groups).argmax() for c in set(labels) - {-1}}
    rows = [
        BucketRow(
            neuron=neuron_idx, mode=mode, bucket=b,
            range_low=corpus[corpus_ids == b].min(), range_high=corpus[corpus_ids == b].max(),
            n_in_corpus=int((corpus_ids == b).sum()), n_kept=int((kept_bucket == b).sum()),
            n_noise=int(((kept_bucket == b) & (labels < 0)).sum()),
            clusters_touching=len(set(labels[(kept_bucket == b) & (labels >= 0)])),
            clusters_homed=sum(1 for h in home.values() if h == b),
        )
        for b in range(n_groups)
    ]
    top = activations.argmax()
    max_note = f"max activation {activations[top]:.3f}, bucket {kept_bucket[top]}, " + (
        "noise" if labels[top] < 0 else f"cluster {labels[top]} (home bucket {home[labels[top]]})"
    )
    return rows, max_note


MATCH_COSINE = 0.8
"""Two clusters count as the same when their mean Hadamard vectors are at least this similar."""


class Clustering(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hdmd: np.ndarray
    labels: np.ndarray
    activations: np.ndarray


class CoverageRow(BaseModel):
    neuron: int
    mode: str
    n_elbow_clusters: int
    n_captured: int
    worst_captured_cosine: float
    missed_elbow_cosines: list[float | None]
    """Cosine of each uncaptured elbow cluster to its partner, None when it got no partner."""
    n_clusters: int
    n_new: int
    n_new_below_elbow: int
    elbow_clusters_closest_pair_cosine: float


def unit_prototypes(clustering: Clustering) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Cluster ids, their unit-norm mean Hadamard vectors, and their mean activation."""
    ids = sorted(set(clustering.labels) - {-1})
    means = np.stack([clustering.hdmd[clustering.labels == c].mean(axis=0) for c in ids])
    mean_activation = np.array([clustering.activations[clustering.labels == c].mean() for c in ids])
    return ids, means / np.linalg.norm(means, axis=1, keepdims=True), mean_activation


def closest_pair_cosine(prototypes: np.ndarray) -> float:
    """How similar two different elbow clusters already are, for judging what a match means."""
    similarity = prototypes @ prototypes.T
    np.fill_diagonal(similarity, -1.0)
    return float(similarity.max())


def coverage(neuron: int, mode: str, elbow: Clustering, other: Clustering, elbow_value: float) -> CoverageRow:
    """Is every elbow cluster matched by its own cluster of `other`, and what else does `other` find.
    The matching is one to one (linear sum assignment on cosine), so a single cluster of `other`
    cannot stand in for two elbow clusters. A pair only counts below `MATCH_COSINE` as a miss,
    and clusters of `other` without a counted partner are new."""
    _, elbow_protos, _ = unit_prototypes(elbow)
    _, other_protos, other_activation = unit_prototypes(other)
    similarity = elbow_protos @ other_protos.T
    elbow_rows, other_cols = linear_sum_assignment(-similarity)
    matched_cosine = np.full(len(elbow_protos), -1.0)  # -1: no partner left, `other` has fewer clusters
    matched_cosine[elbow_rows] = similarity[elbow_rows, other_cols]
    captured = matched_cosine >= MATCH_COSINE
    partnered = np.zeros(len(other_protos), dtype=bool)
    partnered[other_cols[captured[elbow_rows]]] = True
    is_new = ~partnered
    return CoverageRow(
        neuron=neuron,
        mode=mode,
        n_elbow_clusters=len(elbow_protos),
        n_captured=int(captured.sum()),
        worst_captured_cosine=float(matched_cosine[captured].min()) if captured.any() else 0.0,
        missed_elbow_cosines=[None if c < 0 else round(float(c), 3) for c in matched_cosine[~captured]],
        n_clusters=len(other_protos),
        n_new=int(is_new.sum()),
        n_new_below_elbow=int((is_new & (other_activation < elbow_value)).sum()),
        elbow_clusters_closest_pair_cosine=closest_pair_cosine(elbow_protos) if len(elbow_protos) > 1 else 0.0,
    )


def markdown_table(results: list[ModeResult] | list[BucketRow] | list[CoverageRow]) -> str:
    columns = list(type(results[0]).model_fields)
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for result in results:
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in result.model_dump().values()]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    config = load_run_config(sys.argv[1:])
    neuron_idxs = config.neuron_idxs
    texts = load_texts(config.text, config.seed)
    adapter = report.load_adapter(config)
    run = prepare_run(config, adapter, texts)
    tokenizer = adapter.tokenizer
    capture = run.capture
    scan = capture.scan()
    token_ids = run.source.to_model_batch(list(run.source.sample_ids())).display_ids.cpu()

    targets = [(neuron, mode) for neuron in neuron_idxs for mode in MODES]
    selections = {}
    for neuron, mode in targets:
        mode_hits = config.hits.model_copy(update=MODES[mode])
        selections[(neuron, mode)] = select_hits(scan, neuron, "positive", mode_hits, config.seed)

    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], scan.valid_mask.shape[1])
    gathered = capture.gather(merged.batch_idx, merged.token_idx)
    weight = layer_getter_for(config)(run.module).weight

    results: list[ModeResult] = []
    bucket_rows: list[BucketRow] = []
    max_notes: list[str] = []
    clusterings: dict[tuple[int, str], Clustering] = {}
    for (neuron, mode), index_map in zip(targets, merged.index_maps):
        selection = selections[(neuron, mode)]
        hdmd = hadamard_from_rows(gathered.inputs[index_map], weight, neuron)
        start = time.perf_counter()
        labels = cluster_labels(
            hdmd, config.clustering.min_cluster_size, config.clustering.selection_method, config.clustering.min_samples
        )
        seconds = time.perf_counter() - start
        elbow_value = selections[(neuron, "elbow")].threshold
        clusterings[(neuron, mode)] = Clustering(
            hdmd=hdmd, labels=labels, activations=kept_activations(scan, neuron, selection)
        )
        if mode != "elbow":
            rows, note = bucket_breakdown(
                neuron, mode, selection, scan.neuron_column(neuron)[..., 0].numpy()[scan.valid_mask.numpy()],
                kept_activations(scan, neuron, selection), labels, config.hits,
            )
            bucket_rows += rows
            max_notes.append(f"neuron {neuron} {mode}: {note}")
        results.append(
            summarise(
                neuron, mode, selection, kept_activations(scan, neuron, selection),
                kept_token_ids(token_ids, selection), labels, seconds, elbow_value, tokenizer,
            )
        )

    coverage_rows = [
        coverage(neuron, mode, clusterings[(neuron, "elbow")], clusterings[(neuron, mode)], selections[(neuron, "elbow")].threshold)
        for neuron in neuron_idxs
        for mode in MODES
        if mode != "elbow"
    ]
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "ablation.json").write_text(
        "[" + ",".join(r.model_dump_json() for r in results) + "]"
    )
    table = markdown_table(results)
    (config.out_dir / "ablation.md").write_text(table + "\n")
    print(table)
    bucket_table = markdown_table(bucket_rows) + "\n\n" + "\n".join(f"- {n}" for n in max_notes)
    (config.out_dir / "buckets.md").write_text(bucket_table + "\n")
    print(bucket_table)
    coverage_table = markdown_table(coverage_rows)
    (config.out_dir / "coverage.md").write_text(coverage_table + "\n")
    print(coverage_table)


if __name__ == "__main__":
    main()
