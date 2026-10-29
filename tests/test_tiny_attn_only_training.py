import numpy as np
import torch

from in_tfm.models.attn_only_2l.model import BOS_TOKEN, TRAINED_CONFIG, TRAINED_TOKENIZER, load_trained_attn_only
from in_tfm.tiny_attn_only_training import (
    ModelShape, TinyAttnOnlyTrainConfig, build_model, encode_stories, next_token_loss, repeated_span_windows, sample_windows,
    save_weights, story_starts, train_bpe, training_windows,
)

STORIES = [f"Once upon a time there was a {animal} who liked to {verb}." for animal in ("cat", "dog", "bird")
           for verb in ("run", "jump", "sing", "play")] * 5


def tiny_config(tmp_path) -> TinyAttnOnlyTrainConfig:
    return TinyAttnOnlyTrainConfig(
        _cli_parse_args=False, vocab_size=300, shape=ModelShape(d_model=32, n_heads=2, head_dim=16, seq_len=16),
        batch_size=8, device="cpu", out_dir=tmp_path,
    )


def test_every_story_starts_with_bos_and_decodes_back():
    tokenizer = train_bpe(STORIES, 300)
    tokens = encode_stories(tokenizer, STORIES)
    bos = tokenizer.token_to_id(BOS_TOKEN)
    assert np.sum(tokens == bos) == len(STORIES)
    first_story = tokens[1:np.flatnonzero(tokens == bos)[1]]
    assert tokenizer.decode(first_story.tolist()) == STORIES[0]


def test_windows_begin_at_a_story_and_carry_one_extra_target():
    tokenizer = train_bpe(STORIES, 300)
    tokens = encode_stories(tokenizer, STORIES)
    bos = tokenizer.token_to_id(BOS_TOKEN)
    windows = sample_windows(tokens, story_starts(tokens, bos, 16), 8, 16, np.random.default_rng(0))
    assert windows.shape == (8, 17)
    assert (windows[:, 0] == bos).all()


def test_a_few_steps_lower_the_loss(tmp_path):
    config = tiny_config(tmp_path)
    tokenizer = train_bpe(STORIES, config.vocab_size)
    tokens = encode_stories(tokenizer, STORIES)
    starts = story_starts(tokens, tokenizer.token_to_id(BOS_TOKEN), 16)
    torch.manual_seed(0)
    model = build_model(config)
    windows = sample_windows(tokens, starts, 8, 16, np.random.default_rng(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = next_token_loss(model, windows).item()
    for _ in range(30):
        optimizer.zero_grad()
        next_token_loss(model, windows).backward()
        optimizer.step()
    assert next_token_loss(model, windows).item() < before - 1.0


def test_the_saved_checkpoint_loads_with_its_own_tokenizer(tmp_path):
    config = tiny_config(tmp_path)
    tokenizer = train_bpe(STORIES, config.vocab_size)
    tokenizer.save(str(tmp_path / TRAINED_TOKENIZER))
    model = build_model(config)
    model.config.to_json_file(tmp_path / TRAINED_CONFIG)
    save_weights(model, tmp_path / "final.pt")
    loaded, hf_tokenizer = load_trained_attn_only(tmp_path)
    ids = torch.tensor([hf_tokenizer("Once upon a time").input_ids])
    assert torch.allclose(loaded(input_ids=ids).logits, model.eval()(input_ids=ids).logits)
    assert hf_tokenizer.bos_token_id == tokenizer.token_to_id(BOS_TOKEN)
    assert all(name.endswith("lm_head.bias") or not name.endswith("bias") for name in loaded.state_dict())


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
