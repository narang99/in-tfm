"""Integrated Gradients for one conv neuron at one output position, through captum.

- It differentiates the same scalar that `deeplift_attribution` selects, so the two are comparable.
- The baseline is zeros, as in `deeplift`.
- Unlike DeepLift it needs `n_steps` forward and backward passes, and only converges to the activation change as they grow.
"""

from collections.abc import Callable

import numpy as np
import torch
from captum.attr import IntegratedGradients, NeuronIntegratedGradients
from jaxtyping import Float, Int
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

from ..layers import LayerGetter
from ..sources import ModelBatch

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


def compute_integrated_gradients_relevance(
    model: torch.nn.Module,
    batch: ModelBatch,
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int | None,
    n_steps: int = 128,
    internal_batch_size: int = 256,
) -> Float[np.ndarray, "batch ..."]:
    """The same explanation as `compute_deeplift_relevance`, through captum's Integrated Gradients.

    - It exists to check DeepLift against a method with no approximation rules, at the cost of `n_steps` passes.
    - The baseline is zeros, as for DeepLift, so both sum to the same activation change when IG has converged.
    """
    if token_idx is None:
        raise ValueError("Integrated Gradients here explains a single output position, pass token_idx")
    model.eval()
    _, leaf = batch.differentiable()
    images = leaf.detach()
    positions = torch.full((len(images),), token_idx, device=images.device)
    forward = explained_activation(model, layer_getter(model), neuron_idx)
    return integrated_gradients(forward, images, positions, n_steps, internal_batch_size).cpu().numpy()


def neuron_ig(
    model: Dinov2Model,
    pixel_values: Float[torch.Tensor, "batch c h w"],
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int,
) -> Float[torch.Tensor, "batch h w"]:
    """Sums over the channel axis before returning, matching compute_attnlrp_relevance's
    contract so callers (e.g. mk_overlay) don't need to care which attribution method produced
    the map.
    """
    forward_fn = lambda px: model(pixel_values=px)
    ig = NeuronIntegratedGradients(forward_fn, layer_getter(model))
    attr = ig.attribute(pixel_values, (token_idx, neuron_idx), internal_batch_size=4)
    return attr.sum(dim=1)
