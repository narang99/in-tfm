"""Per-hit Hadamard products, ready for clustering.

A layer's input explains a neuron only after it is weighted by the neuron's weight row, element-wise.
The product is `input * weight[neuron_idx]`, an exact decomposition of the activation.
"""

import numpy as np
import torch
from jaxtyping import Float


def near_square_shape(n: int) -> tuple[int, int]:
    """(h, w) with h * w == n and h <= w, as close to square as n's factors allow - a display
    shape for a flat vector that has no natural 2D layout. A prime n degrades to (1, n)."""
    h = int(np.sqrt(n))
    while n % h:
        h -= 1
    return h, n // h


def raw_hadamard_from_rows(
    rows: Float[torch.Tensor, "n_hits hidden"],
    weight: Float[torch.Tensor, "out_hidden hidden"],
    neuron_idx: int,
) -> Float[np.ndarray, "n_hits hidden"]:
    return (rows.detach().cpu() * weight[neuron_idx].detach().cpu()).clone().numpy()
