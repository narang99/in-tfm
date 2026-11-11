from in_tfm.sources.wikitext import articles, paragraphs

LONG = "word " * 60


def test_paragraphs_drop_headings_and_short_rows():
    rows = [
        f" = Title = \n",
        "",
        "short",
        f"  {LONG}  \n",
        f" = = Section = = \n",
        LONG + "again",
    ]
    assert paragraphs(rows) == [LONG.strip(), LONG + "again"]


def test_articles_start_at_level_one_headings_and_keep_section_headings():
    rows = [
        " = First = \n",
        "a\n",
        " = = Section = = \n",
        "b\n",
        " = Second = \n",
        "c\n",
    ]
    assert articles(rows) == [
        "",
        "= First = \na\n = = Section = = \nb",
        "= Second = \nc",
    ]


def test_a_document_corpus_keeps_every_non_empty_row(monkeypatch):
    import datasets

    from in_tfm.run_config import TextDataConfig
    from in_tfm.sources.wikitext import load_texts

    rows = {"text": ["first doc", "  ", "second doc"]}
    monkeypatch.setattr(datasets, "load_dataset", lambda name, config, split: rows)
    texts = load_texts(
        TextDataConfig(
            dataset="some/corpus", dataset_config=None, unit="document", n_samples=10
        ),
        seed=0,
    )
    assert sorted(texts) == ["first doc", "second doc"]


def test_a_mix_splits_its_samples_by_share_and_rounds_to_the_largest():
    from in_tfm.run_config import MixedTextPart
    from in_tfm.sources.wikitext import samples_per_part

    parts = [
        MixedTextPart(dataset="a", share=4),
        MixedTextPart(dataset="b", share=1),
        MixedTextPart(dataset="c", share=1),
    ]
    assert samples_per_part(parts, 15000) == [10000, 2500, 2500]
    assert samples_per_part(parts, 10) == [8, 1, 1]


class FakeStream:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def shuffle(self, seed: int, buffer_size: int) -> "FakeStream":
        return self

    def __iter__(self):
        return iter(self.rows)


def test_a_mix_streams_each_part_from_its_own_column(monkeypatch):
    import datasets

    from in_tfm.run_config import TextDataConfig
    from in_tfm.sources.wikitext import load_texts

    corpora = {
        "web": FakeStream(
            [{"text": "w1"}, {"text": " "}, {"text": "w2"}, {"text": "w3"}]
        ),
        "code": FakeStream([{"content": "c1"}, {"content": "c2"}]),
    }
    monkeypatch.setattr(
        datasets, "load_dataset", lambda name, config, split, streaming: corpora[name]
    )
    part = {"unit": "document", "streaming": True}
    texts = load_texts(
        TextDataConfig(
            n_samples=3,
            mix=[
                {"dataset": "web", "share": 2, **part},
                {"dataset": "code", "column": "content", **part},
            ],
        ),
        seed=0,
    )
    assert sorted(texts) == ["c1", "w1", "w2"]
