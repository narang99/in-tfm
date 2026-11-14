import torch
from transformers import AutoTokenizer

from in_tfm.sources import TextSource


def test_a_text_hit_s_sentence_stops_at_sentence_ends_and_skips_left_padding():
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer.padding_side = "left"
    text = "Tom went to the park. He saw a big red ball! He was happy."
    source = TextSource([text], tokenizer, torch.nn.Embedding(50257, 4), 32)
    n_tokens = len(tokenizer(text)["input_ids"])
    column = 32 - n_tokens + 8
    assert source.sentences_around(["0"], [column]) == [
        (" He saw", " a", " big red ball!")
    ]
