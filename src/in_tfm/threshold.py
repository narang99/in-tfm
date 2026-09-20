"""Automatic per-neuron activation threshold via elbow detection, replacing a hand-picked
constant (see embs-v2.ipynb's manually eyeballed THRESH=0.9) with a repeatable cutoff: most
tokens barely activate a given neuron, then a small tail rises sharply, and the elbow marks
where that tail starts.
"""

import typing

import torch
from jaxtyping import Float


def find_elbow_index_in_sorted_data(data: torch.Tensor) -> int:
    """Kneedle-style knee point: index of max perpendicular distance from the chord connecting
    the first and last points of a sorted curve."""
    n = len(data)
    x = torch.arange(n, dtype=data.dtype, device=data.device)
    y = data

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    numerator = ((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1).abs()
    denominator = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator

    return typing.cast(int, torch.argmax(distances).item())


def positive_sorted_activations(
    outputs: Float[torch.Tensor, "batch seq hidden"], neuron_idx: int
) -> Float[torch.Tensor, "n_pos"]:
    values = outputs[..., neuron_idx].flatten()
    values = values[values > 0]
    return torch.sort(values).values


def find_activation_threshold(
    outputs: Float[torch.Tensor, "batch seq hidden"], neuron_idx: int
) -> tuple[float, Float[torch.Tensor, "n_pos"], int]:
    sorted_values = positive_sorted_activations(outputs, neuron_idx)
    elbow_idx = find_elbow_index_in_sorted_data(sorted_values)
    return sorted_values[elbow_idx].item(), sorted_values, elbow_idx
