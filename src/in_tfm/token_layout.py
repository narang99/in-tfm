"""How a layer's tensors map onto the `(batch, seq, hidden)` shape the pipeline works in.

- The model's adapter picks the layout for a layer, see `ModelAdapter.layout_for`.
- A linear layer in a sequence model already has that shape, so `SequenceLayout` passes it through.
- A linear layer after a flatten has no position axis, so `SingleTokenLayout` adds one of length 1.
  - Each sample is then a single position, and a hit is a whole sample.
- Each layout checks the number of dims it was given, so a layer given the wrong layout raises.
- A conv is a linear map over unfolded patches, so it can be viewed the same way.
  - `seq` is the output grid flattened row by row, so `token_idx = y * width + x`.
  - The input becomes the patch each output position reads, flattened to `C_in * k * k`.
  - The output becomes the channels at each position.
  - The weight flattens to `(C_out, C_in * k * k)`, so `patch * weight[neuron]` is the Hadamard product.
- The conversion runs on saved tensors after capture, so the model's own forward is untouched.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from jaxtyping import Float


@runtime_checkable
class TokenLayout(Protocol):
    def inputs(
        self, layer_input: torch.Tensor
    ) -> Float[torch.Tensor, "batch seq hidden"]: ...

    def outputs(
        self, layer_output: torch.Tensor
    ) -> Float[torch.Tensor, "batch seq out_hidden"]: ...

    def weight(
        self, layer_weight: torch.Tensor
    ) -> Float[torch.Tensor, "out_hidden hidden"]: ...


def require_ndim(tensor: torch.Tensor, ndim: int, layout: str) -> None:
    if tensor.ndim != ndim:
        raise ValueError(
            f"{layout} expects {ndim}-d tensors, got shape {tuple(tensor.shape)}"
        )


class SequenceLayout:
    def inputs(
        self, layer_input: Float[torch.Tensor, "batch seq hidden"]
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        require_ndim(layer_input, 3, "SequenceLayout")
        return layer_input

    def outputs(
        self, layer_output: Float[torch.Tensor, "batch seq out_hidden"]
    ) -> Float[torch.Tensor, "batch seq out_hidden"]:
        require_ndim(layer_output, 3, "SequenceLayout")
        return layer_output

    def weight(
        self, layer_weight: Float[torch.Tensor, "out_hidden hidden"]
    ) -> Float[torch.Tensor, "out_hidden hidden"]:
        return layer_weight


class SingleTokenLayout:
    def inputs(
        self, layer_input: Float[torch.Tensor, "batch hidden"]
    ) -> Float[torch.Tensor, "batch 1 hidden"]:
        require_ndim(layer_input, 2, "SingleTokenLayout")
        return layer_input.unsqueeze(1)

    def outputs(
        self, layer_output: Float[torch.Tensor, "batch out_hidden"]
    ) -> Float[torch.Tensor, "batch 1 out_hidden"]:
        require_ndim(layer_output, 2, "SingleTokenLayout")
        return layer_output.unsqueeze(1)

    def weight(
        self, layer_weight: Float[torch.Tensor, "out_hidden hidden"]
    ) -> Float[torch.Tensor, "out_hidden hidden"]:
        return layer_weight


class ConvLayout:
    def __init__(self, conv: torch.nn.Conv2d) -> None:
        if conv.groups != 1:
            raise ValueError(
                "grouped convs have no single patch to take a Hadamard product against"
            )
        if isinstance(conv.padding, str):
            raise ValueError(
                "padding given as a string has no fixed patch grid to unfold"
            )
        self.kernel_size = conv.kernel_size
        self.dilation = conv.dilation
        self.padding = conv.padding
        self.stride = conv.stride

    def inputs(
        self, layer_input: Float[torch.Tensor, "batch c_in h w"]
    ) -> Float[torch.Tensor, "batch seq patch"]:
        patches = F.unfold(
            layer_input,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        return patches.transpose(1, 2)

    def outputs(
        self, layer_output: Float[torch.Tensor, "batch c_out h w"]
    ) -> Float[torch.Tensor, "batch seq c_out"]:
        return layer_output.flatten(2).transpose(1, 2)

    def weight(
        self, layer_weight: Float[torch.Tensor, "c_out c_in kh kw"]
    ) -> Float[torch.Tensor, "c_out patch"]:
        return layer_weight.flatten(1)


def image_layer_layout(layer: torch.nn.Module) -> TokenLayout:
    """For the image models, where every `Linear` comes after a flatten."""
    if isinstance(layer, torch.nn.Conv2d):
        return ConvLayout(layer)
    if isinstance(layer, torch.nn.Linear):
        return SingleTokenLayout()
    raise ValueError(f"no token layout for a {type(layer).__name__} in an image model")
