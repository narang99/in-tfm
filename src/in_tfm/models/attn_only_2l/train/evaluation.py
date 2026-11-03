import numpy as np
import torch
from jaxtyping import Int
from pydantic import BaseModel

from ..model import AttnOnlyForCausalLM
from .config import TinyAttnOnlyTrainConfig
from .loss import next_token_loss
from .windows import sample_windows


class EvalPoint(BaseModel):
    step: int
    tokens: int
    train_loss: float
    val_loss: float
    seconds: float

    def summary(self, total_steps: int) -> str:
        return (
            f"[{self.step}/{total_steps}] tokens {self.tokens:,} train {self.train_loss:.3f} "
            f"val {self.val_loss:.3f} {self.seconds:.0f}s"
        )


@torch.no_grad()
def val_loss(
    model: AttnOnlyForCausalLM,
    tokens: Int[np.ndarray, "n_tokens"],
    starts: Int[np.ndarray, "n_starts"],
    config: TinyAttnOnlyTrainConfig,
    rng: np.random.Generator,
) -> float:
    losses = [
        next_token_loss(
            model,
            sample_windows(
                tokens, starts, config.batch_size, config.shape.seq_len, rng
            ).to(config.device),
        ).item()
        for _ in range(config.eval_batches)
    ]
    return float(np.mean(losses))


def evaluate(
    model: AttnOnlyForCausalLM,
    train_loss: float,
    step: int,
    seen: int,
    val_tokens: Int[np.ndarray, "n_tokens"],
    val_starts: Int[np.ndarray, "n_starts"],
    config: TinyAttnOnlyTrainConfig,
    rng: np.random.Generator,
    seconds: float,
) -> EvalPoint:
    model.eval()
    with torch.autocast(
        "cuda", dtype=torch.float16, enabled=config.device.startswith("cuda")
    ):
        loss = val_loss(model, val_tokens, val_starts, config, rng)
    model.train()
    return EvalPoint(
        step=step, tokens=seen, train_loss=train_loss, val_loss=loss, seconds=seconds
    )
