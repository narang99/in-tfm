"""Walks back from the logits, keeping each neuron's top-share terms.

A neuron's kept inputs depend only on the neuron, never on which logit reached it. So every
neuron is expanded once and shared by all the logits that reach it.
"""

from typing import Annotated, Literal

import torch
from jaxtyping import Bool
from pydantic import BaseModel, ConfigDict, Field

from .flow import Terms, TermsOf, Unembed
from .neurons import Neuron, PosEmbed, Resid, ResidMid, TokenEmbed
from .top_share import top_share_mask


class AllLogits(BaseModel):
    kind: Literal["all"] = "all"

    def select(self, seq: int, vocab: int) -> Bool[torch.Tensor, "seq vocab"]:
        return torch.ones(seq, vocab, dtype=torch.bool)


LogitSelector = Annotated[AllLogits, Field(discriminator="kind")]


class UsefulNeurons(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logit_picks: Bool[torch.Tensor, "seq vocab hidden"]
    """Which final resid dims each logit kept. Stored as a tensor, since there are vocab x seq logits."""
    picks: dict[Neuron, list[Neuron]]
    """Every other picked neuron, mapped to the inputs it kept. Leaves map to an empty list."""


def pick_logit_inputs(
    unembed: Unembed, selector: LogitSelector, share: float
) -> Bool[torch.Tensor, "seq vocab hidden"]:
    seq, vocab = unembed.inputs.shape[0], unembed.W_U.shape[0]
    selected = selector.select(seq, vocab)
    picks = []
    for pos in range(seq):
        hadamard = unembed.W_U * unembed.inputs[pos]
        kept = top_share_mask(hadamard, share) & selected[pos, :, None]
        picks.append(kept)
    return torch.stack(picks)


def pick_inputs(neuron: Neuron, terms_of: TermsOf, share: float) -> list[Neuron]:
    terms = terms_of(neuron)
    if terms is None:
        return []
    kept = top_share_mask(terms.values, share).nonzero().flatten().tolist()
    return [input for i in kept for input in terms.inputs_of(i)]


def walk_back(
    roots: list[Neuron], terms_of: TermsOf, share: float
) -> dict[Neuron, list[Neuron]]:
    """Equal neurons are interned to one object, since edge lists would otherwise hold millions of copies."""
    picks: dict[Neuron, list[Neuron]] = {}
    interned: dict[Neuron, Neuron] = {}
    frontier = list(roots)
    while frontier:
        neuron = frontier.pop()
        if neuron in picks:
            continue
        inputs = pick_inputs(neuron, terms_of, share)
        picks[neuron] = [interned.setdefault(input, input) for input in inputs]
        frontier.extend(picks[neuron])
    return picks


def find_useful_neurons(
    unembed: Unembed, terms_of: TermsOf, selector: LogitSelector, share: float
) -> UsefulNeurons:
    logit_picks = pick_logit_inputs(unembed, selector, share)
    picked_resid = logit_picks.any(dim=1).nonzero().tolist()
    roots: list[Neuron] = [
        Resid(layer=unembed.final_layer, pos=pos, dim=dim) for pos, dim in picked_resid
    ]
    picks = walk_back(roots, terms_of, share)
    return UsefulNeurons(logit_picks=logit_picks, picks=picks)


STREAM_KINDS = (Resid, ResidMid, TokenEmbed, PosEmbed)


def within_blocks(terms_of: TermsOf) -> TermsOf:
    """Stops at the residual stream, so a walk never leaves the block it started in."""

    def block_terms(neuron: Neuron) -> Terms | None:
        if isinstance(neuron, STREAM_KINDS):
            return None
        return terms_of(neuron)

    return block_terms


def find_block_neurons(
    writers: list[Neuron], terms_of: TermsOf, share: float
) -> dict[Neuron, list[Neuron]]:
    return walk_back(writers, within_blocks(terms_of), share)
