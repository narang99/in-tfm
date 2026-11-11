"""Walks back from the logits, keeping each neuron's top-share terms.

A neuron's kept inputs depend only on the neuron, never on which logit reached it. So every
neuron is expanded once and shared by all the logits that reach it.
"""

from typing import Annotated, Literal

import torch
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict, Field

from .neurons import Neuron, Resid
from .terms import terms_of
from .top_share import top_share_mask
from .trace import Trace


class AllLogits(BaseModel):
    kind: Literal["all"] = "all"

    def select(self, trace: Trace) -> Bool[torch.Tensor, "seq vocab"]:
        seq, vocab = trace.tokens.shape[0], trace.W_U.shape[0]
        return torch.ones(seq, vocab, dtype=torch.bool)


LogitSelector = Annotated[AllLogits, Field(discriminator="kind")]


class UsefulNeurons(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logit_picks: Bool[torch.Tensor, "seq vocab hidden"]
    """Which final resid dims each logit kept. Stored as a tensor, since there are vocab x seq logits."""
    picks: dict[Neuron, list[Neuron]]
    """Every other picked neuron, mapped to the inputs it kept. Leaves map to an empty list."""


def pick_logit_inputs(
    trace: Trace, selector: LogitSelector, share: float
) -> Bool[torch.Tensor, "seq vocab hidden"]:
    selected = selector.select(trace)
    final_resid = trace.resid[-1]
    picks = []
    for pos in range(final_resid.shape[0]):
        hadamard = trace.W_U * final_resid[pos]
        kept = top_share_mask(hadamard, share) & selected[pos, :, None]
        picks.append(kept)
    return torch.stack(picks)


def pick_inputs(neuron: Neuron, trace: Trace, share: float) -> list[Neuron]:
    terms = terms_of(neuron, trace)
    if terms is None:
        return []
    kept = top_share_mask(terms.values, share).nonzero().flatten().tolist()
    return [input for i in kept for input in terms.inputs_of(i)]


def walk_back(
    roots: list[Neuron], trace: Trace, share: float
) -> dict[Neuron, list[Neuron]]:
    picks: dict[Neuron, list[Neuron]] = {}
    frontier = list(roots)
    while frontier:
        neuron = frontier.pop()
        if neuron in picks:
            continue
        picks[neuron] = pick_inputs(neuron, trace, share)
        frontier.extend(picks[neuron])
    return picks


def find_useful_neurons(
    trace: Trace, selector: LogitSelector, share: float
) -> UsefulNeurons:
    logit_picks = pick_logit_inputs(trace, selector, share)
    picked_resid = logit_picks.any(dim=1).nonzero().tolist()
    roots: list[Neuron] = [
        Resid(layer=trace.final_layer, pos=pos, dim=dim) for pos, dim in picked_resid
    ]
    return UsefulNeurons(logit_picks=logit_picks, picks=walk_back(roots, trace, share))
