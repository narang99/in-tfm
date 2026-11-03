"""The tokenizer and the encoded corpus.

- The tokenizer is a byte-level BPE trained on the corpus itself.
  - A small vocab makes token pairs repeat far more often, which is what the clusters need.
- The corpus becomes one token stream, with BOS before each story.
"""

import numpy as np
from datasets import load_dataset
from jaxtyping import Int
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ..models.attn_only_2l.model import BOS_TOKEN, TRAINED_TOKENIZER
from .config import TOKENS_FILE, TinyAttnOnlyTrainConfig


def train_bpe(stories: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[BOS_TOKEN], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(stories, trainer)
    return tokenizer


def encode_stories(tokenizer: Tokenizer, stories: list[str], chunk: int = 20_000) -> Int[np.ndarray, "n_tokens"]:
    bos = tokenizer.token_to_id(BOS_TOKEN)
    pieces = []
    for start in range(0, len(stories), chunk):
        for encoding in tokenizer.encode_batch(stories[start:start + chunk]):
            pieces.append(np.array([bos, *encoding.ids], dtype=np.uint16))
    return np.concatenate(pieces)


def story_starts(tokens: Int[np.ndarray, "n_tokens"], bos: int, seq_len: int) -> Int[np.ndarray, "n_starts"]:
    """Each story's BOS, dropping those too close to the end to fit a whole window."""
    starts = np.flatnonzero(tokens == bos)
    return starts[starts + seq_len + 1 <= len(tokens)]


def prepare_tokens(config: TinyAttnOnlyTrainConfig) -> tuple[Tokenizer, dict[str, np.ndarray]]:
    """Cached in `out_dir`, so a second run skips the tokenizer and the encoding."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = config.out_dir / TRAINED_TOKENIZER
    splits = {"train": "train", "val": "validation"}
    cached = [config.out_dir / TOKENS_FILE.format(split=name) for name in splits]
    if tokenizer_path.exists() and all(path.exists() for path in cached):
        return Tokenizer.from_file(str(tokenizer_path)), {name: np.load(path) for name, path in zip(splits, cached)}
    corpus = load_dataset(config.dataset)
    stories = {name: corpus[split][config.text_column] for name, split in splits.items()}
    tokenizer = train_bpe(stories["train"][:config.tokenizer_stories], config.vocab_size)
    tokenizer.save(str(tokenizer_path))
    arrays = {name: encode_stories(tokenizer, texts) for name, texts in stories.items()}
    for path, array in zip(cached, arrays.values()):
        np.save(path, array)
    return tokenizer, arrays
