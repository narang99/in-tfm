"""Turning raw neuron activations into per-hit Hadamard products, ready for clustering.

fc2's input is the neuron's own "why did this fire" signal only after being weighted by the
neuron's row of the fc2 weight matrix (element-wise) - the Hadamard product below is exactly
that: fc2.input * fc2.weight[neuron_idx]. It's L2-normalized because clustering (HDBSCAN
downstream) should group hits by which input dimensions drove the activation, not by how
strongly the neuron fired.
"""

import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn.preprocessing import normalize


def near_square_shape(n: int) -> tuple[int, int]:
    """(h, w) with h * w == n and h <= w, as close to square as n's factors allow - a display
    shape for a flat vector that has no natural 2D layout. A prime n degrades to (1, n)."""
    h = int(np.sqrt(n))
    while n % h:
        h -= 1
    return h, n // h


def hadamard_products(
    inputs: Float[torch.Tensor, "batch seq hidden"],
    batch_idx: Int[np.ndarray, "n_hits"],
    token_idx: Int[np.ndarray, "n_hits"],
    weight: Float[torch.Tensor, "out_hidden hidden"],
    neuron_idx: int,
) -> Float[np.ndarray, "n_hits hidden"]:
    return hadamard_from_rows(inputs[batch_idx, token_idx], weight, neuron_idx)


def hadamard_from_rows(
    rows: Float[torch.Tensor, "n_hits hidden"],
    weight: Float[torch.Tensor, "out_hidden hidden"],
    neuron_idx: int,
) -> Float[np.ndarray, "n_hits hidden"]:
    """For inputs already gathered at the hit positions - see neuron_capture.NeuronCapture,
    which never holds the full (batch, seq, hidden) input."""
    return normalize(raw_hadamard_from_rows(rows, weight, neuron_idx), "l2")


def raw_hadamard_from_rows(
    rows: Float[torch.Tensor, "n_hits hidden"],
    weight: Float[torch.Tensor, "out_hidden hidden"],
    neuron_idx: int,
) -> Float[np.ndarray, "n_hits hidden"]:
    return (rows.detach().cpu() * weight[neuron_idx].detach().cpu()).clone().numpy()

