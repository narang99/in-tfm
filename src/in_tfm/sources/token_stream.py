"""A corpus that is already tokenized, read as fixed windows of one long token array.

- A scan of hundreds of millions of positions cannot hold every text as a Python string.
  - The stream is memory-mapped, so only the windows of the current batch are read.
- Windows start at a story's BOS and do not overlap, as in `attn_only_2l.train`.
- Every window is `seq_len` real tokens, so a whole scan has no padding.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Integer
from pydantic import BaseModel, ConfigDict

from .base import SampleId
from .text import TextSource


class TokenStream(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: Integer[np.ndarray, "n_tokens"]
    max_windows: int


def load_token_stream(path: Path, max_windows: int) -> TokenStream:
    return TokenStream(tokens=np.load(path, mmap_mode="r"), max_windows=max_windows)


def story_aligned_starts(
    stream: Integer[np.ndarray, "n_tokens"], bos: int, seq_len: int, max_windows: int
) -> Integer[np.ndarray, "n_windows"]:
    """Greedy: each window starts at the first story start past the end of the previous window."""
    stories = np.flatnonzero(stream == bos)
    stories = stories[stories + seq_len <= len(stream)]
    picked = []
    next_free = 0
    for start in stories.tolist():
        if start >= next_free:
            picked.append(start)
            next_free = start + seq_len
            if len(picked) == max_windows:
                break
    return np.array(picked, dtype=np.int64)


class TokenStreamSource(TextSource):
    def __init__(
        self,
        stream: TokenStream,
        tokenizer,
        embed: torch.nn.Module,
        seq_len: int,
        bos_is_a_hit: bool = True,
    ) -> None:
        super().__init__([], tokenizer, embed, seq_len, False, bos_is_a_hit)
        self.stream = stream.tokens
        self.starts = story_aligned_starts(
            stream.tokens, tokenizer.bos_token_id, seq_len, stream.max_windows
        )

    def sample_ids(self) -> Sequence[SampleId]:
        return [str(i) for i in range(len(self.starts))]

    def text_for(self, sample_id: SampleId) -> str:
        return self.tokenizer.decode(self._window(sample_id))

    def _window(self, sample_id: SampleId) -> Integer[np.ndarray, "seq"]:
        start = self.starts[int(sample_id)]
        return np.asarray(self.stream[start : start + self.max_length], dtype=np.int64)

    def _encode(self, ids: Sequence[SampleId]):
        input_ids = torch.from_numpy(np.stack([self._window(i) for i in ids]))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
