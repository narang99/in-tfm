import torch
from jaxtyping import Int
from torch import nn

from ..model import AttnOnlyForCausalLM


def next_token_loss(model: AttnOnlyForCausalLM, windows: Int[torch.Tensor, "batch seq_plus_one"]) -> torch.Tensor:
    logits = model(input_ids=windows[:, :-1]).logits
    return nn.functional.cross_entropy(logits.flatten(0, 1).float(), windows[:, 1:].flatten())
