#!/usr/bin/env python
"""Fake cached payload for the text presenter, so its html can be iterated on with no model.

Writes `meta.json`, `elbow.png` and per-cluster `hits.pt` in the same layout `write_report`
caches, then calls `render_report`. Re-run after every presenter edit:

    uv run python scripts/fake_text_report.py

The stub tokenizer's ids only mean something inside this process, so this script always writes
fresh fake data. `render_report` itself works on any real report folder that has a `hits.pt`.
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from in_tfm.neuron_report import (
    ClusterMeta,
    ReportMeta,
    cluster_dir_for,
    render_report,
    save_hits,
)
from in_tfm.presenters import ClusterHit, TextPresenter
from in_tfm.viz import save_elbow_plot

HADAMARD_SHAPE = (8, 16)
HIDDEN = HADAMARD_SHAPE[0] * HADAMARD_SHAPE[1]
BOS = "<bos>"

FAKE_CLUSTERS: dict[int, list[str]] = {
    3: [
        "On January 15 , *1861* , the state legislature decided to secede",
        "were issued M1816 / M1822 . *69* caliber flint locks",
        "it was named one of the top *10* attractions in the state",
        "On April *12* , 1893 the tower building and the surrounding grounds",
        "the . *69* caliber smoothbores and shotguns remained the predominant weapon",
    ],
    1: [
        "Most of the equipment , arms *,* and machinery at the Arsenal was removed",
        "a collection point for armaments *and* ammunition manufacture for small arms",
        "four officers : himself *,* Major John B. Lockman , Captain C.C. Pennington",
        "the best in the series . His main criticisms were its length *and* gameplay repetition",
    ],
    2: [
        "has assumed such an aspect that *it* becomes my duty as the executive",
        "Imca and Riela . *They* were Senjō no Valkyria 3 : Nam",
        "were hired . *She* spent much time in bed at home amusing herself",
    ],
}
"""Cluster id -> sentences, whitespace-tokenized, with the firing word wrapped in `*`.

Digits are split one per token and words get a leading SentencePiece space, like Gemma."""


class StubTokenizer:
    """Just enough of a HF tokenizer for `TextPresenter`: ids in, token strings out."""

    def __init__(self) -> None:
        self.vocab: list[str] = []

    def encode_word(self, word: str, first: bool) -> list[str]:
        prefix = "" if first else "▁"
        if word.isdigit():
            return [prefix + word[0]] + list(word[1:])
        return [prefix + word]

    def add(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab.append(token)
        return self.vocab.index(token)

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self.vocab[int(i)] for i in ids]


class StubSource:
    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()


def fake_hit(
    source: StubSource, sentence: str, sample_id: str, cluster_pattern: np.ndarray
) -> ClusterHit:
    tokens = [BOS]
    firing_token_idx = 0
    for i, word in enumerate(sentence.split()):
        is_firing = word.startswith("*") and word.endswith("*") and len(word) > 2
        pieces = source.tokenizer.encode_word(word.strip("*") if is_firing else word, first=(i == 0))
        tokens.extend(pieces)
        if is_firing:
            firing_token_idx = len(tokens) - 1

    rng = np.random.default_rng(zlib.crc32(sentence.encode()))
    per_token = rng.normal(0, 0.03, len(tokens))
    per_token[0] = -rng.uniform(0.5, 1.5)  # the <bos> attention sink
    per_token[firing_token_idx] = rng.uniform(0.2, 0.5)
    for offset in (-1, -2):
        if firing_token_idx + offset > 0:
            per_token[firing_token_idx + offset] = rng.uniform(0.0, 0.25)

    # spread each token's number over the hidden axis; the presenter sums it back
    weights = rng.dirichlet(np.ones(HIDDEN), size=len(tokens))
    relevance = (per_token[:, None] * weights)[None].astype(np.float32)
    return ClusterHit(
        sample_id=sample_id,
        token_idx=firing_token_idx,
        relevance=relevance,
        model_input=torch.zeros(1),  # the text presenter never reads it
        hadamard=(cluster_pattern + 0.5 * rng.normal(size=HIDDEN)).astype(np.float32),
        display_ids=torch.tensor([source.tokenizer.add(t) for t in tokens]),
    )


def write_fake_cache(report_dir: Path, source: StubSource) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_ids: list[str] = []
    clusters: dict[int, ClusterMeta] = {}
    for cid, examples in FAKE_CLUSTERS.items():
        hits = []
        # hits in a cluster share a pattern, as real clustered Hadamard vectors do
        cluster_pattern = np.random.default_rng(cid).normal(size=HIDDEN)
        for sentence in examples:
            sample_id = str(len(sample_ids))
            sample_ids.append(sample_id)
            hits.append(fake_hit(source, sentence, sample_id, cluster_pattern))
        cluster_dir = cluster_dir_for(report_dir, cid)
        cluster_dir.mkdir(exist_ok=True)
        save_hits(hits, cluster_dir)
        clusters[cid] = ClusterMeta(n_hits=len(hits) * 12, n_unique_images=len(hits))

    meta = ReportMeta(
        neuron_idx=90,
        threshold=0.0809,
        n_hits=sum(c.n_hits for c in clusters.values()),
        min_uniq_images_per_cluster=2,
        n_clusters=len(clusters),
        hadamard_shape=HADAMARD_SHAPE,
        clusters=clusters,
        sample_ids=sample_ids,
    )
    (report_dir / "meta.json").write_text(meta.model_dump_json(indent=2))
    elbow = np.sort(np.random.default_rng(0).exponential(0.2, 400))[::-1].copy()
    save_elbow_plot(elbow, 40, report_dir / "elbow.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports_fake/neuron_90"))
    args = parser.parse_args()

    source = StubSource()
    write_fake_cache(args.report_dir, source)
    presenter = TextPresenter(source)
    print(render_report(args.report_dir, presenter))


if __name__ == "__main__":
    main()
