"""Each neuron's mean activation over a corpus, stored as `PersistedValues`.

Attention weights get 0, since a weight belongs to a (token, source) pair, not to a neuron.
"""

import torch

from .replacement import PersistedValues, Site

SiteActivations = dict[Site, tuple[torch.Tensor, int]]
"""Each site's activation, and which axis of it is the sequence."""


class MeanAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.tokens = 0

    def add(self, activations: SiteActivations, n_tokens: int) -> None:
        for site, (activation, seq_axis) in activations.items():
            summed = activation.sum(dim=seq_axis, keepdim=True)
            self.sums[site.key] = self.sums.get(site.key, 0) + summed
        self.tokens += n_tokens

    def means(self, n_layers: int) -> PersistedValues:
        values = {key: total / self.tokens for key, total in self.sums.items()}
        for layer in range(n_layers):
            values[Site(name="pattern", layer=layer).key] = torch.tensor(0.0)
        return PersistedValues(values=values)
