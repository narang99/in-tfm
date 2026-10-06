from in_tfm.presenters.text.colored_tokens import display_text


def test_byte_level_bpe_markers_become_the_characters_they_stand_for():
    assert display_text("Ġthe") == " the"
    assert display_text("Ċ") == "↵"
    assert display_text("aĉb") == "a\tb"


def test_sentencepiece_space_still_works():
    assert display_text("▁the") == " the"
