"""Integrated Gradients for one conv neuron at one output position, through captum.

- It differentiates the same scalar that `deeplift_attribution` selects, so the two are comparable.
- The baseline is zeros, as in `deeplift`.
- Unlike DeepLift it needs `n_steps` forward and backward passes, and only converges to the activation change as they grow.
"""

from collections.abc import Callable

import torch
from captum.attr import IntegratedGradients
from jaxtyping import Float, Int

Images = Float[torch.Tensor, "batch c h w"]
Positions = Int[torch.Tensor, "batch"]
Attribution = Float[torch.Tensor, "batch c h w"]


def pick_position(layer_output: torch.Tensor, neuron: int, positions: Positions) -> Float[torch.Tensor, "batch"]:
    per_position = layer_output[:, neuron].flatten(1)
    return per_position[torch.arange(len(positions), device=positions.device), positions]


def explained_activation(
    model: torch.nn.Module, layer: torch.nn.Module, neuron: int
) -> Callable[[Images, Positions], Float[torch.Tensor, "batch"]]:
    """What Integrated Gradients differentiates, the activation at each sample's own position."""

    def forward(images: Images, positions: Positions) -> Float[torch.Tensor, "batch"]:
        seen: list[torch.Tensor] = []
        handle = layer.register_forward_hook(lambda m, i, o: seen.append(pick_position(o, neuron, positions)))
        try:
            model(images)
        finally:
            handle.remove()
        return seen[0]

    return forward


def integrated_gradients(
    forward: Callable[[Images, Positions], torch.Tensor],
    images: Images,
    positions: Positions,
    n_steps: int,
    internal_batch_size: int,
) -> Attribution:
    ig = IntegratedGradients(forward)
    return ig.attribute(
        images,
        baselines=torch.zeros_like(images),
        additional_forward_args=(positions,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    ).detach()
