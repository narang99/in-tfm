"""What every model hands the walk: how a neuron splits into terms, and what the unembed reads."""

from collections.abc import Callable
from typing import NamedTuple

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

from .neurons import Neuron


class Terms(NamedTuple):
    values: Float[torch.Tensor, " n"]
    inputs_of: Callable[[int], tuple[Neuron, ...]]
    """Built lazily, so only the kept terms ever create neurons."""


TermsOf = Callable[[Neuron], Terms | None]


class Unembed(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    W_U: Float[torch.Tensor, "vocab hidden"]
    inputs: Float[torch.Tensor, "seq hidden"]
    """What logit t's Hadamard multiplies `W_U[t]` with: the final resid, after any frozen final norm."""
    final_layer: int
