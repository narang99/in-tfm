"""A corpus mixed from several datasets, each part streamed and optionally filtered by a regex.

- Streaming, since corpora like FineWeb are far larger than the documents one run reads.
- Each stream is shuffled through a buffer first, so a part is not the first files of one shard.
"""

import random
import re
from collections.abc import Iterable
from itertools import islice

from ..run_config import TextPart

SHUFFLE_BUFFER = 10_000


def compiled(pattern: str | None) -> re.Pattern[str] | None:
    return None if pattern is None else re.compile(pattern, re.MULTILINE)


def wanted_texts(texts: Iterable[str], part: TextPart) -> list[str]:
    matching = compiled(part.matching)
    not_matching = compiled(part.not_matching)
    wanted = (
        text
        for text in texts
        if text.strip()
        and (matching is None or matching.search(text))
        and (not_matching is None or not not_matching.search(text))
    )
    return list(islice(wanted, part.n_samples))


def load_part(part: TextPart, seed: int) -> list[str]:
    from datasets import load_dataset

    stream = load_dataset(
        part.dataset, part.dataset_config, split=part.split, streaming=True
    ).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    texts = wanted_texts((row[part.column] for row in stream), part)
    if len(texts) < part.n_samples:
        raise ValueError(
            f"{part.dataset} ran out at {len(texts)} of {part.n_samples} wanted texts"
        )
    return texts


def load_mix(parts: list[TextPart], seed: int) -> list[str]:
    texts = []
    for part in parts:
        loaded = load_part(part, seed)
        print(f"{len(loaded)} texts from {part.dataset} matching {part.matching!r}")
        texts.extend(loaded)
    random.Random(seed).shuffle(texts)
    return texts
