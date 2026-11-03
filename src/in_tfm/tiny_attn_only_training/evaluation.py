import numpy as np
import torch
from jaxtyping import Int
from pydantic import BaseModel

from ..models.attn_only_2l.model import AttnOnlyForCausalLM
from .config import TinyAttnOnlyTrainConfig
from .induction import induction_check
from .loss import next_token_loss
from .windows import sample_windows


class EvalPoint(BaseModel):
    step: int
    tokens: int
    train_loss: float
    val_loss: float
    """On plain windows only."""
    induction_first: float
    induction_second: float
    """Loss on the second copy of a repeated random sequence. Far below the first copy means induction works."""
    induction_attention: float
    """The best layer 1 head's attention from the second copy to the token after the same token in the first copy."""
    previous_attention: float
    """The best layer 0 head's attention to the previous token, on the same sequences."""
    seconds: float

    def summary(self, total_steps: int) -> str:
        return (f"[{self.step}/{total_steps}] tokens {self.tokens:,} train {self.train_loss:.3f} "
                f"val {self.val_loss:.3f} induction {self.induction_first:.2f}->{self.induction_second:.2f} "
                f"attn {self.induction_attention:.2f} previous {self.previous_attention:.2f} {self.seconds:.0f}s")


@torch.no_grad()
def val_loss(model: AttnOnlyForCausalLM, tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"],
             config: TinyAttnOnlyTrainConfig, rng: np.random.Generator) -> float:
    losses = [
        next_token_loss(model, sample_windows(tokens, starts, config.batch_size, config.shape.seq_len, rng)
                        .to(config.device)).item()
        for _ in range(config.eval_batches)
    ]
    return float(np.mean(losses))


def evaluate(model: AttnOnlyForCausalLM, train_loss: float, step: int, seen: int, bos: int,
             val_tokens: Int[np.ndarray, "n_tokens"], val_starts: Int[np.ndarray, "n_starts"],
             config: TinyAttnOnlyTrainConfig, rng: np.random.Generator, seconds: float) -> EvalPoint:
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=config.device.startswith("cuda")):
        loss = val_loss(model, val_tokens, val_starts, config, rng)
        check = induction_check(model, bos, config, rng)
    model.train()
    return EvalPoint(step=step, tokens=seen, train_loss=train_loss, val_loss=loss, induction_first=check.loss_first,
                     induction_second=check.loss_second, induction_attention=check.induction_attention,
                     previous_attention=check.previous_attention, seconds=seconds)
