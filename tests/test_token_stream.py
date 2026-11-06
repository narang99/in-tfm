import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from in_tfm.sources.token_stream import (
    TokenStream,
    TokenStreamSource,
    story_aligned_starts,
)

BOS = 0
STREAM = np.array([0, 5, 6, 0, 7, 0, 8, 9, 10, 0, 11, 12, 0, 13], dtype=np.uint16)


def fake_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<bos>": BOS, "<unk>": 1}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        bos_token="<bos>",
        unk_token="<unk>",
    )


def test_windows_start_at_a_story_and_do_not_overlap():
    assert story_aligned_starts(STREAM, BOS, 4, 100).tolist() == [0, 5, 9]


def test_a_window_past_the_end_is_dropped():
    assert 12 not in story_aligned_starts(STREAM, BOS, 4, 100).tolist()


def test_max_windows_stops_at_the_first_ones():
    assert story_aligned_starts(STREAM, BOS, 4, 2).tolist() == [0, 5]


def test_a_batch_has_real_tokens_and_no_padding():
    embed = torch.nn.Embedding(20, 3)
    source = TokenStreamSource(
        TokenStream(tokens=STREAM, max_windows=100), fake_tokenizer(), embed, 4
    )
    assert source.sample_ids() == ["0", "1", "2"]
    batch = source.to_model_batch(["1", "2"])
    assert batch.display_ids is not None
    assert batch.display_ids.tolist() == [[0, 8, 9, 10], [0, 11, 12, 0]]
    assert batch.valid_mask.all()
    assert batch.kwargs["inputs_embeds"].shape == (2, 4, 3)
