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


def test_a_part_keeps_matching_texts_up_to_its_count():
    from in_tfm.run_config import TextPart
    from in_tfm.sources.text_mix import wanted_texts

    texts = [
        "import os\nimport tornado.web",
        "",
        "from tornado import gen",
        "import flask",
        "from tornado.ioloop import IOLoop",
    ]
    part = TextPart(
        dataset="some/code",
        n_samples=2,
        matching=r"^\s*(import tornado|from tornado[\s.])",
    )
    assert wanted_texts(texts, part) == texts[0:1] + texts[2:3]


def test_a_part_drops_texts_its_exclusion_finds():
    from in_tfm.run_config import TextPart
    from in_tfm.sources.text_mix import wanted_texts

    part = TextPart(dataset="some/code", n_samples=5, not_matching=r"\btornado\b")
    assert wanted_texts(["import tornado", "import flask"], part) == ["import flask"]
