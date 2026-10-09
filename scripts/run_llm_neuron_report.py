#!/usr/bin/env python
"""Neuron reports for a decoder language model, using the same pipeline as the vision path.

Everything between activation capture and clustering is shared with run_neuron_report.py - only
the source (text instead of DICOMs), the presenter (tokens in context instead of pixel overlays)
and the layer getter (down_proj instead of fc2) differ. See in_tfm.sources for why text has to
hand the model embeddings rather than token ids.

Clusters here group tokens by *which intermediate MLP neurons drove the activation*, so a
cluster is a claim about what kind of language the neuron responds to - the report prints the
firing token for each cluster so that claim is readable at a glance.

Settings are in `in_tfm.run_config.RunConfig`: defaults come from `--config <yaml>` (see
`configs/`), and flags override them. `--help` lists every flag.
"""

import random
import re
import sys
import time

from in_tfm.models.attn_only_2l import AttnOnly2LAdapter
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.models.text import DecoderTextAdapter
from in_tfm.neuron_run import run_neuron_reports, timed
from in_tfm.run_config import ModelName, RunConfig, TextDataConfig, load_run_config

WIKITEXT_TITLE = re.compile(r"^ = [^=].* = \n?$")
"""A level-1 heading; the `= =` of a section heading fails the `[^=]` after the first `= `."""


def paragraphs(rows: list[str]) -> list[str]:
    """Non-empty paragraphs only - wikitext is full of blank lines and bare `= Heading =`
    markers, which tokenize to almost nothing and would pad out every batch."""
    stripped = [row.strip() for row in rows]
    return [text for text in stripped if len(text) > 200 and not text.startswith("=")]


def articles(rows: list[str]) -> list[str]:
    """Rows are single paragraphs; a level-1 heading starts a new article."""
    joined: list[list[str]] = [[]]
    for row in rows:
        if WIKITEXT_TITLE.match(row):
            joined.append([])
        joined[-1].append(row)
    return ["".join(parts).strip() for parts in joined]


def load_texts(data: TextDataConfig, seed: int) -> list[str]:
    """Shuffled before truncating to `n_samples`: the corpus is in file order, so taking the
    first N would be N paragraphs of the same few articles."""
    from datasets import load_dataset

    rows = list(load_dataset(data.dataset, data.dataset_config, split=data.split)["text"])
    if data.unit == "paragraph":
        texts = paragraphs(rows)
    else:
        texts = [a for a in articles(rows) if len(a) >= data.min_article_chars]
    random.Random(seed).shuffle(texts)
    return texts[: data.n_samples]


def load_adapter(config: RunConfig) -> DecoderTextAdapter:
    if config.model is ModelName.ATTN_ONLY_2L:
        return AttnOnly2LAdapter.from_pretrained(config.data.max_length)
    return Gemma3Adapter.from_pretrained(config.model.value, config.data.max_length)


def main() -> None:
    pipeline_start = time.perf_counter()
    config = load_run_config(sys.argv[1:])

    with timed("load texts"):
        texts = load_texts(config.data, config.seed)
    print(f"loaded {len(texts)} {config.data.unit}s")

    with timed("load model"):
        adapter = load_adapter(config)

    run_neuron_reports(config, adapter, texts)
    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
