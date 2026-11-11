"""A text corpus as a list of texts: wikitext by paragraph or by article, or any corpus by document.

A mixed corpus loads each of its datasets the same way, then shuffles them together.
"""

import random
import re
from itertools import islice

from ..run_config import MixedTextPart, TextDataConfig, TextDataset

SHUFFLE_BUFFER = 10_000

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


def texts_from_rows(rows: list[str], dataset: TextDataset) -> list[str]:
    match dataset.unit:
        case "paragraph":
            return paragraphs(rows)
        case "article":
            return [a for a in articles(rows) if len(a) >= dataset.min_article_chars]
        case "document":
            return [row for row in rows if row.strip()]


def downloaded_texts(dataset: TextDataset, n_samples: int, seed: int) -> list[str]:
    """Shuffled before truncating to `n_samples`: the corpus is in file order, so taking the
    first N would be N paragraphs of the same few articles."""
    from datasets import load_dataset

    rows = list(
        load_dataset(dataset.dataset, dataset.dataset_config, split=dataset.split)[
            dataset.column
        ]
    )
    texts = texts_from_rows(rows, dataset)
    random.Random(seed).shuffle(texts)
    return texts[:n_samples]


def streamed_texts(dataset: TextDataset, n_samples: int, seed: int) -> list[str]:
    """Shuffled through a buffer first, so the texts are not the first rows of one shard."""
    from datasets import load_dataset

    stream = load_dataset(
        dataset.dataset, dataset.dataset_config, split=dataset.split, streaming=True
    ).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    documents = (row[dataset.column] for row in stream)
    texts = list(islice((text for text in documents if text.strip()), n_samples))
    if len(texts) < n_samples:
        raise ValueError(
            f"{dataset.dataset} ran out at {len(texts)} of {n_samples} texts"
        )
    return texts


def dataset_texts(dataset: TextDataset, n_samples: int, seed: int) -> list[str]:
    if dataset.streaming:
        return streamed_texts(dataset, n_samples, seed)
    return downloaded_texts(dataset, n_samples, seed)


def samples_per_part(parts: list[MixedTextPart], n_samples: int) -> list[int]:
    """Flooring leaves a few samples over, which go to the part with the largest share."""
    total_share = sum(part.share for part in parts)
    counts = [int(n_samples * part.share / total_share) for part in parts]
    largest = max(range(len(parts)), key=lambda i: parts[i].share)
    counts[largest] += n_samples - sum(counts)
    return counts


def mixed_texts(parts: list[MixedTextPart], n_samples: int, seed: int) -> list[str]:
    texts = []
    for part, count in zip(parts, samples_per_part(parts, n_samples)):
        loaded = dataset_texts(part, count, seed)
        print(f"{len(loaded)} texts from {part.dataset}")
        texts.extend(loaded)
    random.Random(seed).shuffle(texts)
    return texts


def load_texts(text_config: TextDataConfig, seed: int) -> list[str]:
    if text_config.mix is not None:
        return mixed_texts(text_config.mix, text_config.n_samples, seed)
    return dataset_texts(text_config, text_config.n_samples, seed)
