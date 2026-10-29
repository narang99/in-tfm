"""Train a small attention-only shortformer model on a corpus we know, so clustering can use that same corpus.

- The architecture is `AttnOnlyForCausalLM`, the shim for `callummcdougall/attn_only_2L_half`.
  - It has no MLP and no normalization, and position goes only into the queries and keys.
  - The attention projections have no bias. The unembed keeps one, which learns the unigram logits.
- The tokenizer is a byte-level BPE trained on the corpus itself.
  - A small vocab makes token pairs repeat far more often, which is what the clusters need.
- Every training window starts at a story's BOS, like the report adapter, which prepends BOS.
- `repeat_share` of the windows repeat a span of story text, so the model has a reason to learn induction.
  - The val loss is on plain windows only.
- The token arrays are saved next to the checkpoint, so a later report reads the exact training stream.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn

from .device import default_device
from .models.attn_only_2l.model import (
    BOS_TOKEN, TRAINED_CONFIG, TRAINED_TOKENIZER, TRAINED_WEIGHTS, AttnOnlyConfig, AttnOnlyForCausalLM, attention_patterns,
)

TOKENS_FILE = "{split}_tokens.npy"
LOG_FILE = "train_log.jsonl"


class ModelShape(BaseModel):
    d_model: int = 256
    n_heads: int = 8
    head_dim: int = 32
    n_layers: int = 2
    seq_len: int = 256


class TinyAttnOnlyTrainConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    dataset: str = "roneneldan/TinyStories"
    text_column: str = "text"
    vocab_size: int = 4096
    tokenizer_stories: int = 200_000
    """How many training stories the BPE merges are learned from. The whole split is then encoded."""
    shape: ModelShape = ModelShape()
    batch_size: int = 128
    lr: float = 1e-3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_tokens: int = 500_000_000
    eval_every: int = 500
    eval_batches: int = 20
    snapshot_tokens: list[int] = [1_000_000, 5_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000]
    """Extra checkpoints along the way, to see when the induction heads form."""
    repeat_share: float = 0.0
    """The share of training windows that repeat a span of story text, see `repeated_span_windows`."""
    seed: int = 0
    device: str = default_device()
    out_dir: Path = Path("checkpoints/tiny_attn_only_2l")


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
    """Each story becomes BOS followed by its tokens, and the stories are joined into one stream."""
    bos = tokenizer.token_to_id(BOS_TOKEN)
    pieces = []
    for start in range(0, len(stories), chunk):
        for encoding in tokenizer.encode_batch(stories[start:start + chunk]):
            pieces.append(np.array([bos, *encoding.ids], dtype=np.uint16))
    return np.concatenate(pieces)


def story_starts(tokens: Int[np.ndarray, "n_tokens"], bos: int, seq_len: int) -> Int[np.ndarray, "n_starts"]:
    starts = np.flatnonzero(tokens == bos)
    return starts[starts + seq_len + 1 <= len(tokens)]


def sample_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], batch_size: int, seq_len: int,
    rng: np.random.Generator,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    """One extra token per window, so the inputs and the next-token targets come from one slice."""
    chosen = rng.choice(starts, size=batch_size)
    return torch.from_numpy(np.stack([tokens[s:s + seq_len + 1] for s in chosen]).astype(np.int64))


def repeated_span_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], batch_size: int, seq_len: int,
    rng: np.random.Generator, min_span: int = 16,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    """BOS, a span of a story, the same span again, then the story goes on.

    - On TinyStories alone a 2-layer model never formed induction heads in 500M tokens.
      - Names are what repeats there, and a single head that attends to earlier names can already copy them.
    - The second copy of a span is predictable only by induction, so these windows pay for the circuit.
    """
    chosen = rng.choice(starts, size=batch_size)
    spans = rng.integers(min(min_span, seq_len // 2), seq_len // 2 + 1, size=batch_size)
    windows = []
    for start, span in zip(chosen, spans):
        text = tokens[start + 1:start + 1 + span]
        rest = tokens[start + 1 + span:start + 1 + seq_len]
        windows.append(np.concatenate([tokens[start:start + 1], text, text, rest])[:seq_len + 1])
    return torch.from_numpy(np.stack(windows).astype(np.int64))


def training_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], config: "TinyAttnOnlyTrainConfig",
    rng: np.random.Generator,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    n_repeat = rng.binomial(config.batch_size, config.repeat_share)
    seq_len = config.shape.seq_len
    parts = [(sample_windows, config.batch_size - n_repeat), (repeated_span_windows, n_repeat)]
    return torch.cat([draw(tokens, starts, n, seq_len, rng) for draw, n in parts if n > 0])


def build_model(config: TinyAttnOnlyTrainConfig) -> AttnOnlyForCausalLM:
    shape = config.shape
    model = AttnOnlyForCausalLM(AttnOnlyConfig(
        hidden_size=shape.d_model, num_attention_heads=shape.n_heads, head_dim=shape.head_dim,
        num_hidden_layers=shape.n_layers, max_position_embeddings=shape.seq_len, vocab_size=config.vocab_size,
        attention_bias=False,
    ))
    init_weights(model, shape.n_layers)
    return model


def init_weights(model: AttnOnlyForCausalLM, n_layers: int) -> None:
    """GPT-2 style. The output projections are scaled down because each one adds to an unnormalized stream."""
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(param)
        elif "o_proj" in name:
            nn.init.normal_(param, std=0.02 / math.sqrt(2 * n_layers))
        else:
            nn.init.normal_(param, std=0.02)


def next_token_loss(model: AttnOnlyForCausalLM, windows: Int[torch.Tensor, "batch seq_plus_one"]) -> torch.Tensor:
    logits = model(input_ids=windows[:, :-1]).logits
    return nn.functional.cross_entropy(logits.flatten(0, 1).float(), windows[:, 1:].flatten())


def lr_at(step: int, total_steps: int, config: TinyAttnOnlyTrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1 + math.cos(math.pi * progress))


class EvalPoint(BaseModel):
    step: int
    tokens: int
    train_loss: float
    val_loss: float
    induction_first: float
    induction_second: float
    """Loss on the second copy of a repeated random sequence. Far below the first copy means induction works."""
    induction_attention: float
    """The best layer 1 head's attention from the second copy to the token after the same token in the first copy."""
    previous_attention: float
    """The best layer 0 head's attention to the previous token, on the same sequences."""
    seconds: float


@torch.no_grad()
def val_loss(model, tokens, starts, config: TinyAttnOnlyTrainConfig, rng: np.random.Generator) -> float:
    losses = [
        next_token_loss(model, sample_windows(tokens, starts, config.batch_size, config.shape.seq_len, rng)
                        .to(config.device)).item()
        for _ in range(config.eval_batches)
    ]
    return float(np.mean(losses))


class InductionCheck(BaseModel):
    loss_first: float
    loss_second: float
    induction_attention: float
    previous_attention: float


@torch.no_grad()
def induction_check(model, bos: int, config: TinyAttnOnlyTrainConfig, rng: np.random.Generator,
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


def save_weights(model: AttnOnlyForCausalLM, path: Path) -> None:
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)


def train(config: TinyAttnOnlyTrainConfig) -> Path:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    tokenizer, arrays = prepare_tokens(config)
    bos = tokenizer.token_to_id(BOS_TOKEN)
    seq_len = config.shape.seq_len
    starts = {name: story_starts(array, bos, seq_len) for name, array in arrays.items()}
    print(f"tokens: train {len(arrays['train']):,}, val {len(arrays['val']):,}, stories {len(starts['train']):,}")

    model = build_model(config).to(config.device)
    model.config.to_json_file(config.out_dir / TRAINED_CONFIG)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95),
                                  weight_decay=config.weight_decay)
    use_fp16 = config.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_fp16)
    tokens_per_step = config.batch_size * seq_len
    total_steps = config.max_tokens // tokens_per_step
    pending_snapshots = sorted(config.snapshot_tokens)
    log = (config.out_dir / LOG_FILE).open("w")
    began = time.time()

    for step in range(total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, config)
        windows = training_windows(arrays["train"], starts["train"], config, rng).to(config.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            loss = next_token_loss(model, windows)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        seen = (step + 1) * tokens_per_step
        while pending_snapshots and seen >= pending_snapshots[0]:
            save_weights(model, config.out_dir / f"tokens_{pending_snapshots.pop(0)}.pt")
        if (step + 1) % config.eval_every == 0 or step + 1 == total_steps:
            point = evaluate(model, loss.item(), step + 1, seen, bos, arrays["val"], starts["val"], config, rng,
                             time.time() - began)
            log.write(point.model_dump_json() + "\n")
            log.flush()
            print(f"[{point.step}/{total_steps}] tokens {seen:,} train {point.train_loss:.3f} val {point.val_loss:.3f} "
                  f"induction {point.induction_first:.2f}->{point.induction_second:.2f} attn {point.induction_attention:.2f} "
                  f"previous {point.previous_attention:.2f} {point.seconds:.0f}s")

    path = config.out_dir / TRAINED_WEIGHTS
    save_weights(model, path)
    (config.out_dir / "train_config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    print(f"saved {path} after {time.time() - began:.0f}s")
    return path


def evaluate(model, train_loss: float, step: int, seen: int, bos: int, val_tokens, val_starts,
             config: TinyAttnOnlyTrainConfig, rng: np.random.Generator, seconds: float) -> EvalPoint:
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=config.device.startswith("cuda")):
        loss = val_loss(model, val_tokens, val_starts, config, rng)
        check = induction_check(model, bos, config, rng)
    model.train()
    return EvalPoint(step=step, tokens=seen, train_loss=train_loss, val_loss=loss, induction_first=check.loss_first,
                     induction_second=check.loss_second, induction_attention=check.induction_attention,
                     previous_attention=check.previous_attention, seconds=seconds)
