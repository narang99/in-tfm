from collections.abc import Sequence

import torch

from ...sources import ModelBatch, SampleId
from .model import EQUALS_TOKEN, SEQ_LEN, GrokTransformer

EMBEDS_KEY = "inputs_embeds"
"""Named after `GrokTransformer.forward`'s argument, which the batch is splatted into."""


def pair_id(a: int, b: int) -> SampleId:
    return f"{a},{b}"


def parse_pair_id(sample_id: SampleId) -> tuple[int, int]:
    a, b = sample_id.split(",")
    return int(a), int(b)


class PairSource:
    """Sums `a + b`, as the sequence `a b =`. The sample id is `a,b`."""

    def __init__(self, pairs: Sequence[tuple[int, int]], model: GrokTransformer) -> None:
        self.pairs = list(pairs)
        self.embed = model.embed_tokens

    def sample_ids(self) -> Sequence[SampleId]:
        return [pair_id(a, b) for a, b in self.pairs]

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        tokens = torch.tensor([[*parse_pair_id(i), EQUALS_TOKEN] for i in ids])
        with torch.no_grad():
            embeds = self.embed(tokens.to(self.embed.weight.device))
        return ModelBatch(
            kwargs={EMBEDS_KEY: embeds},
            grad_leaf_key=EMBEDS_KEY,
            valid_mask=torch.ones(len(ids), SEQ_LEN, dtype=torch.bool),
            display_ids=tokens,
        )
