import numpy as np

from in_tfm.models.attn_only_2l.model import BOS_TOKEN
from in_tfm.tiny_attn_only_training import (
    ModelShape, TinyAttnOnlyTrainConfig, encode_stories, random_span, repeated_span_windows, story_starts, train_bpe,
    training_windows,
)

STORIES = [f"Once upon a time there was a {animal} who liked to {verb}." for animal in ("cat", "dog", "bird")
           for verb in ("run", "jump", "sing", "play")] * 5


def tiny_config(tmp_path) -> TinyAttnOnlyTrainConfig:
    return TinyAttnOnlyTrainConfig(
        _cli_parse_args=False, vocab_size=300, shape=ModelShape(d_model=32, n_heads=2, head_dim=16, seq_len=16),
        batch_size=8, device="cpu", out_dir=tmp_path,
    )


def test_a_repeated_window_holds_its_span_twice_after_bos():
    tokens = np.concatenate([[0], np.arange(1, 200), [0], np.arange(200, 400)]).astype(np.uint16)
    starts = story_starts(tokens, 0, 32)
    windows = repeated_span_windows(tokens, starts, 4, 32, np.random.default_rng(0), min_span=16)
    assert windows.shape == (4, 33)
    for window in windows.numpy():
        assert window[0] == 0
        span = window[1:17]
        assert np.array_equal(window[17:33], span)


def test_a_batch_of_only_repeats_or_only_plain_windows_still_draws(tmp_path):
    tokenizer = train_bpe(STORIES, 300)
    tokens = encode_stories(tokenizer, STORIES)
    starts = story_starts(tokens, tokenizer.token_to_id(BOS_TOKEN), 16)
    for share in (0.0, 1.0):
        config = tiny_config(tmp_path).model_copy(update={"repeat_share": share})
        assert training_windows(tokens, starts, config, np.random.default_rng(0)).shape == (8, 17)


def test_a_random_span_repeats_and_never_draws_bos():
    tokens = np.concatenate([[0], np.arange(1, 200), [0], np.arange(200, 400)]).astype(np.uint16)
    starts = story_starts(tokens, 0, 32)
    windows = repeated_span_windows(tokens, starts, 64, 32, np.random.default_rng(0), min_span=16,
                                    span_tokens=random_span(0, 5))
    assert np.array_equal(windows[:, 17:33], windows[:, 1:17])
    assert set(np.unique(windows[:, 1:17].numpy())) <= {1, 2, 3, 4}
