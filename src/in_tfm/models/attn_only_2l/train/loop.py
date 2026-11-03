"""The training loop: AdamW, warmup then cosine decay, gradient clipping, and fp16 autocast on CUDA."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..model import TRAINED_CONFIG, TRAINED_WEIGHTS, AttnOnlyForCausalLM
from .config import LOG_FILE, TinyAttnOnlyTrainConfig
from .corpus import bos_id, prepare_tokens, story_starts
from .evaluation import evaluate
from .loss import next_token_loss
from .model_setup import build_model
from .windows import sample_windows


def train(config: TinyAttnOnlyTrainConfig) -> Path:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    tokenizer, arrays = prepare_tokens(config)
    bos = bos_id(tokenizer)
    seq_len = config.shape.seq_len
    starts = {name: story_starts(array, bos, seq_len) for name, array in arrays.items()}
    print(
        f"tokens: train {len(arrays['train']):,}, val {len(arrays['val']):,}, stories {len(starts['train']):,}"
    )

    model = build_model(config).to(config.device)
    model.config.to_json_file(config.out_dir / TRAINED_CONFIG)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    use_fp16 = config.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_fp16)
    tokens_per_step = config.batch_size * seq_len
    total_steps = config.max_tokens // tokens_per_step
    pending_snapshots = sorted(config.snapshot_tokens)
    log = (config.out_dir / LOG_FILE).open("w")
    began = time.time()

    for step in range(total_steps):
        set_lr(optimizer, lr_at(step, total_steps, config))
        windows = sample_windows(
            arrays["train"], starts["train"], config.batch_size, seq_len, rng
        ).to(config.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            loss = next_token_loss(model, windows)
        optimizer_step(model, optimizer, scaler, loss)

        seen = (step + 1) * tokens_per_step
        save_due_snapshots(model, pending_snapshots, seen, config.out_dir)
        if (step + 1) % config.eval_every == 0 or step + 1 == total_steps:
            point = evaluate(
                model,
                loss.item(),
                step + 1,
                seen,
                arrays["val"],
                starts["val"],
                config,
                rng,
                time.time() - began,
            )
            log.write(point.model_dump_json() + "\n")
            log.flush()
            print(point.summary(total_steps))

    path = config.out_dir / TRAINED_WEIGHTS
    save_weights(model, path)
    (config.out_dir / "train_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2)
    )
    print(f"saved {path} after {time.time() - began:.0f}s")
    return path


def lr_at(step: int, total_steps: int, config: TinyAttnOnlyTrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_step(
    model: AttnOnlyForCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss: torch.Tensor,
) -> None:
    """Gradients are unscaled before clipping, so the clip sees their true norm under fp16."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


def save_due_snapshots(
    model: AttnOnlyForCausalLM, pending: list[int], seen: int, out_dir: Path
) -> None:
    while pending and seen >= pending[0]:
        save_weights(model, out_dir / f"tokens_{pending.pop(0)}.pt")


def save_weights(model: AttnOnlyForCausalLM, path: Path) -> None:
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}, path
    )
