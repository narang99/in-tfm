from in_tfm.text_corpus import articles, paragraphs

LONG = "word " * 60


def test_paragraphs_drop_headings_and_short_rows():
    rows = [f" = Title = \n", "", "short", f"  {LONG}  \n", f" = = Section = = \n", LONG + "again"]
    assert paragraphs(rows) == [LONG.strip(), LONG + "again"]


def test_articles_start_at_level_one_headings_and_keep_section_headings():
    rows = [" = First = \n", "a\n", " = = Section = = \n", "b\n", " = Second = \n", "c\n"]
    assert articles(rows) == ["", "= First = \na\n = = Section = = \nb", "= Second = \nc"]
