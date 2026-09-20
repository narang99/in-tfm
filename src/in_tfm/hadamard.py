"""Turning raw neuron activations into per-hit Hadamard products, ready for clustering.

fc2's input is the neuron's own "why did this fire" signal only after being weighted by the
neuron's row of the fc2 weight matrix (element-wise) - the Hadamard product below is exactly
that: fc2.input * fc2.weight[neuron_idx]. It's L2-normalized because clustering (HDBSCAN
downstream) should group hits by which input dimensions drove the activation, not by how
strongly the neuron fired.
"""

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from sklearn.preprocessing import normalize


def high_activation_hits(
    outputs: Float[torch.Tensor, "batch seq hidden"],
    neuron_idx: int,
    thresh: float,
    valid_mask: Bool[torch.Tensor, "batch seq"] | None = None,
) -> tuple[Int[np.ndarray, "n_hits"], Int[np.ndarray, "n_hits"]]:
    """`valid_mask` excludes padding. Ragged batches (text) pad to the longest sequence, and a
    pad position can sit above the threshold like any other - those hits cluster perfectly
    happily and mean nothing, so they have to be dropped before `argwhere`, not after.
    """
    token_activation = outputs[..., neuron_idx].detach().cpu().numpy()  # (batch, seq_len)
    above = token_activation > thresh
    if valid_mask is not None:
        above &= valid_mask.detach().cpu().numpy()
    hits = np.argwhere(above)  # (num_hits, 2): (batch_idx, token_idx)
    return hits[:, 0], hits[:, 1]


def hadamard_products(
    inputs: Float[torch.Tensor, "batch seq hidden"],
    batch_idx: Int[np.ndarray, "n_hits"],
    token_idx: Int[np.ndarray, "n_hits"],
    weight: Float[torch.Tensor, "out_hidden hidden"],
    neuron_idx: int,
) -> Float[np.ndarray, "n_hits hidden"]:
    pats = inputs[batch_idx, token_idx]  # (num_hits, hidden)
    hdmd = pats * weight[neuron_idx]
    hdmd = hdmd.detach().cpu().clone().numpy()
    return normalize(hdmd, "l2")
