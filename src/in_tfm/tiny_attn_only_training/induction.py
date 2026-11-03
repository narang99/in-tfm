"""What makes induction heads form in a model this small, and the check that they did.

- On TinyStories alone a 2-layer model never formed induction heads in 500M tokens.
  - Names are what repeats there, and a single head that attends to earlier names can already copy them.
- So some training windows repeat a span, whose second copy only induction can predict.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from jaxtyping import Int
from pydantic import BaseModel
from torch import nn

from ..models.attn_only_2l.model import AttnOnlyForCausalLM, attention_patterns
from .config import TinyAttnOnlyTrainConfig
from .windows import sample_windows

SpanTokens = Callable[[Int[np.ndarray, "span"], np.random.Generator], Int[np.ndarray, "span"]]


def story_span(text: Int[np.ndarray, "span"], _rng: np.random.Generator) -> Int[np.ndarray, "span"]:
    return text


def random_span(bos: int, vocab_size: int) -> SpanTokens:
    """A span of random tokens, never BOS, which only induction can predict on its second copy."""

    def draw(text: Int[np.ndarray, "span"], rng: np.random.Generator) -> Int[np.ndarray, "span"]:
        drawn = rng.integers(0, vocab_size - 1, size=len(text))
        return (drawn + (drawn >= bos)).astype(text.dtype)

    return draw


def repeated_span_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], batch_size: int, seq_len: int,
    rng: np.random.Generator, min_span: int = 16, span_tokens: SpanTokens = story_span,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    """BOS, a span, the same span again, then the story goes on.

    - `span_tokens` replaces the story's span, for example with random tokens.
    """
    chosen = rng.choice(starts, size=batch_size)
    spans = rng.integers(min(min_span, seq_len // 2), seq_len // 2 + 1, size=batch_size)
    windows = []
    for start, span in zip(chosen, spans):
        text = span_tokens(tokens[start + 1:start + 1 + span], rng)
        rest = tokens[start + 1 + span:start + 1 + seq_len]
        windows.append(np.concatenate([tokens[start:start + 1], text, text, rest])[:seq_len + 1])
    return torch.from_numpy(np.stack(windows).astype(np.int64))


class InductionCheck(BaseModel):
    loss_first: float
    loss_second: float
    induction_attention: float
    previous_attention: float


@torch.no_grad()
def induction_check(model: AttnOnlyForCausalLM, bos: int, config: TinyAttnOnlyTrainConfig, rng: np.random.Generator,
                    batch: int = 32) -> InductionCheck:
    """Random tokens are unpredictable the first time and fully predictable the second time, by copying.

    - The attention scores show the circuit even while the copied token's logit is still too weak to win.
    - Layer 1 is only checked when the model has one.
    """
    length = min(50, (config.shape.seq_len - 1) // 2)
    half = rng.integers(1, config.vocab_size, size=(batch, length))
    windows = torch.from_numpy(np.concatenate([np.full((batch, 1), bos), half, half], axis=1)).to(config.device)
    logits = model(input_ids=windows[:, :-1]).logits.float()
    per_token = nn.functional.cross_entropy(logits.transpose(1, 2), windows[:, 1:], reduction="none")
    every = torch.arange(1, 2 * length + 1, device=config.device)
    previous = attention_patterns(model, windows, 0)[:, :, every, every - 1].mean(dim=(0, 2)).max().item()
    induction = 0.0
    if config.shape.n_layers > 1:
        second = torch.arange(length + 1, 2 * length + 1, device=config.device)
        induction = attention_patterns(model, windows, 1)[:, :, second, second - length + 1].mean(dim=(0, 2)).max().item()
    return InductionCheck(loss_first=per_token[:, 1:length].mean().item(),
                          loss_second=per_token[:, length + 1:].mean().item(),
                          induction_attention=induction, previous_attention=previous)


def training_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], config: TinyAttnOnlyTrainConfig,
    rng: np.random.Generator,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    """A batch mixing plain windows with repeated story spans and repeated random spans, by the config's shares."""
    n_text, n_random, n_plain = rng.multinomial(
        config.batch_size, [config.repeat_share, config.random_repeat_share,
                            1 - config.repeat_share - config.random_repeat_share])
    seq_len = config.shape.seq_len
    random_tokens = random_span(int(tokens[starts[0]]), config.vocab_size)
    parts = [
        (sample_windows, n_plain),
        (repeated_span_windows, n_text),
        (partial(repeated_span_windows, span_tokens=random_tokens), n_random),
    ]
    return torch.cat([draw(tokens, starts, n, seq_len, rng) for draw, n in parts if n > 0])
