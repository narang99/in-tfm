"""The wikitext corpus as a list of texts, one per paragraph or per article."""

import random
import re

from .run_config import TextDataConfig


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


def load_texts(text_config: TextDataConfig, seed: int) -> list[str]:
    """Shuffled before truncating to `n_samples`: the corpus is in file order, so taking the
    first N would be N paragraphs of the same few articles."""
    from datasets import load_dataset

    rows = list(load_dataset(text_config.dataset, text_config.dataset_config, split=text_config.split)["text"])
    if text_config.unit == "paragraph":
        texts = paragraphs(rows)
    else:
        texts = [a for a in articles(rows) if len(a) >= text_config.min_article_chars]
    random.Random(seed).shuffle(texts)
    return texts[: text_config.n_samples]
