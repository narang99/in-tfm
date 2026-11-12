import torch
from jaxtyping import Float, Int
from pydantic import BaseModel

from ..models.attn_only_2l.model import AttnOnlyForCausalLM
from .trace import Trace

Logits = Float[torch.Tensor, "seq vocab"]


class LogitShift(BaseModel):
    """Per position. The next-token lists are one shorter, since the last token has no next."""

    next_token_prob_before: list[float]
    next_token_prob_after: list[float]
    kl: list[float]
    top_token_before: list[int]
    top_token_after: list[int]

    @property
    def percent_change(self) -> list[float]:
        return [
            100 * (after - before) / before
            for before, after in zip(
                self.next_token_prob_before, self.next_token_prob_after
            )
        ]


def trace_logits(trace: Trace, model: AttnOnlyForCausalLM) -> Logits:
    return trace.resid[-1] @ trace.W_U.T + model.lm_head.bias


def kl_per_position(before: Logits, after: Logits) -> Float[torch.Tensor, " seq"]:
    """KL(before || after): how far the ablated distribution moved, weighted by the original's mass."""
    log_p = before.log_softmax(dim=-1)
    log_q = after.log_softmax(dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def next_token_probs(
    logits: Logits, tokens: Int[torch.Tensor, " seq"]
) -> Float[torch.Tensor, " next"]:
    probs = logits[:-1].softmax(dim=-1)
    return probs.gather(-1, tokens[1:, None]).squeeze(-1)


def logit_shift(
    before: Logits, after: Logits, tokens: Int[torch.Tensor, " seq"]
) -> LogitShift:
    return LogitShift(
        next_token_prob_before=next_token_probs(before, tokens).tolist(),
        next_token_prob_after=next_token_probs(after, tokens).tolist(),
        kl=kl_per_position(before, after).tolist(),
        top_token_before=before.argmax(dim=-1).tolist(),
        top_token_after=after.argmax(dim=-1).tolist(),
    )
