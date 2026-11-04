#!/usr/bin/env python
"""Per-cluster token summaries of a neuron report, from every hit rather than the 10 shown.

- Hits and rows are rebuilt with the report's own selection and seed.
- Labels are the hard labels in `hdbscan_soft/soft.npz`, checked against the report's cluster sizes.
- Rows, labels and token ids are saved to `<out_dir>/hits_cache/`, so later steps skip the rebuild.

    python scripts/the_logit_cluster_summary.py --config configs/tiny_attn_only_1l_the_logit.yaml
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from jaxtyping import Int

sys.path.insert(0, str(Path(__file__).parent))
from run_neuron_report import load_adapter, load_samples  # noqa: E402

from in_tfm.hit_selection import select_hits  # noqa: E402
from in_tfm.neuron_run import (  # noqa: E402
    clustered_rows,
    layer_getter_for,
    prepare_run,
    report_name,
)
from in_tfm.run_config import RunConfig, load_run_config  # noqa: E402

PAD = -1


def window_ids(
    texts: list[str], tokenizer, max_length: int
) -> Int[np.ndarray, "n_texts max_length"]:
    """What the report's text source feeds the model: BOS first, cut to `max_length`, padded with PAD."""
    encoded = tokenizer(
        [tokenizer.bos_token + text for text in texts],
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    ids = np.full((len(texts), max_length), PAD, dtype=np.int64)
    for row, seq in enumerate(encoded):
        ids[row, : len(seq)] = seq
    return ids


def build_cache(config: RunConfig, cache: Path) -> None:
    [neuron_idx] = config.neuron_idxs
    adapter = load_adapter(config)
    texts = load_samples(config)
    run = prepare_run(config, adapter, texts)
    scan = run.capture.scan()
    selection = select_hits(scan, neuron_idx, config.polarity, config.hits, config.seed)
    gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)
    rows = clustered_rows(gathered.inputs, weight, neuron_idx, config.clustering)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "rows.npy", rows.astype(np.float32))
    np.save(
        cache / "window_ids.npy",
        window_ids(texts, adapter.tokenizer, config.text.max_length),
    )
    np.savez(
        cache / "hits.npz",
        batch_idx=selection.batch_idx,
        token_idx=selection.token_idx,
        activation=selection.activations,
    )


def check_labels_match_report(labels: Int[np.ndarray, "n"], report_dir: Path) -> None:
    meta = json.loads((report_dir / "meta.json").read_text())
    expected = {int(cid): c["n_hits"] for cid, c in meta["clusters"].items()}
    found = Counter(labels[labels >= 0].tolist())
    mismatched = [cid for cid, n in expected.items() if found.get(cid) != n]
    print(
        f"{len(expected)} report clusters, {len(mismatched)} with a different size in soft.npz"
    )
    assert not mismatched, mismatched[:10]


def top(ids: Int[np.ndarray, "n"], decode, k: int = 6) -> list[tuple[str, float]]:
    return [
        (decode(i), round(n / len(ids), 3))
        for i, n in Counter(ids.tolist()).most_common(k)
    ]


def summaries(
    cache: Path, labels: Int[np.ndarray, "n"], tokenizer, token_id: int
) -> list[dict]:
    hits = np.load(cache / "hits.npz")
    ids = np.load(cache / "window_ids.npy")
    current = ids[hits["batch_idx"], hits["token_idx"]]
    next_idx = np.minimum(hits["token_idx"] + 1, ids.shape[1] - 1)
    following = np.where(
        hits["token_idx"] + 1 < ids.shape[1], ids[hits["batch_idx"], next_idx], PAD
    )
    vocab = {i: tokenizer.decode([i]) for i in range(len(tokenizer))} | {PAD: "<end>"}
    rows = []
    for label in np.unique(labels[labels >= 0]):
        member = labels == label
        rows.append(
            {
                "label": int(label),
                "size": int(member.sum()),
                "positive_share": round(
                    float((hits["activation"][member] > 0).mean()), 3
                ),
                "mean_logit": round(float(hits["activation"][member].mean()), 2),
                "next_is_token": round(
                    float((following[member] == token_id).mean()), 3
                ),
                "median_position": int(np.median(hits["token_idx"][member])),
                "current": top(current[member], vocab.get),
                "next": top(following[member], vocab.get),
            }
        )
    return sorted(rows, key=lambda row: -row["size"])


def main() -> None:
    config = load_run_config(sys.argv[1:])
    [neuron_idx] = config.neuron_idxs
    out = Path(config.out_dir)
    cache = out / "hits_cache"
    if not (cache / "rows.npy").exists():
        build_cache(config, cache)
    labels = np.load(out / "hdbscan_soft" / "soft.npz")["hard"]
    check_labels_match_report(labels, out / report_name(neuron_idx, config.polarity))
    np.save(cache / "labels.npy", labels)
    tokenizer = load_adapter(config).tokenizer
    rows = summaries(cache, labels, tokenizer, neuron_idx)
    (out / "cluster_summary.json").write_text(json.dumps(rows, indent=1))
    print(f"[done] {len(rows)} clusters -> {out / 'cluster_summary.json'}")


if __name__ == "__main__":
    main()
