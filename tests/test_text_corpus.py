from in_tfm.sources.wikitext import articles, paragraphs

LONG = "word " * 60


def test_paragraphs_drop_headings_and_short_rows():
    rows = [f" = Title = \n", "", "short", f"  {LONG}  \n", f" = = Section = = \n", LONG + "again"]
    assert paragraphs(rows) == [LONG.strip(), LONG + "again"]


def test_articles_start_at_level_one_headings_and_keep_section_headings():
    rows = [" = First = \n", "a\n", " = = Section = = \n", "b\n", " = Second = \n", "c\n"]
    assert articles(rows) == ["", "= First = \na\n = = Section = = \nb", "= Second = \nc"]


def test_a_document_corpus_keeps_every_non_empty_row(monkeypatch):
    import datasets

    from in_tfm.run_config import TextDataConfig
    from in_tfm.sources.wikitext import load_texts

    rows = {"text": ["first doc", "  ", "second doc"]}
    monkeypatch.setattr(datasets, "load_dataset", lambda name, config, split: rows)
    texts = load_texts(TextDataConfig(dataset="some/corpus", dataset_config=None, unit="document", n_samples=10), seed=0)
    assert sorted(texts) == ["first doc", "second doc"]
