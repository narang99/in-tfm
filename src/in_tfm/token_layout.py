"""How a layer's tensors map onto the `(batch, seq, hidden)` shape the pipeline works in.

- A linear layer already has that shape, so its layout is the identity.
- A conv is a linear map over unfolded patches, so it can be viewed the same way.
  - `seq` is the output grid flattened row by row, so `token_idx = y * width + x`.
  - The input becomes the patch each output position reads, flattened to `C_in * k * k`.
  - The output becomes the channels at each position.
  - The weight flattens to `(C_out, C_in * k * k)`, so `patch * weight[neuron]` is the Hadamard product.
- The conversion runs on saved tensors after capture, so the model's own forward is untouched.
"""

from typing import Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float


class TokenLayout(Protocol):
    def inputs(self, layer_input: torch.Tensor) -> Float[torch.Tensor, "batch seq hidden"]: ...

    def outputs(self, layer_output: torch.Tensor) -> Float[torch.Tensor, "batch seq out_hidden"]: ...

    def weight(self, layer_weight: torch.Tensor) -> Float[torch.Tensor, "out_hidden hidden"]: ...


class IdentityLayout:
    def inputs(self, layer_input: torch.Tensor) -> torch.Tensor:
        return layer_input

    def outputs(self, layer_output: torch.Tensor) -> torch.Tensor:
        return layer_output

    def weight(self, layer_weight: torch.Tensor) -> torch.Tensor:
        return layer_weight


class ConvLayout:
    def __init__(self, conv: torch.nn.Conv2d) -> None:
        if conv.groups != 1:
            raise ValueError("grouped convs have no single patch to take a Hadamard product against")
        self.kernel_size = conv.kernel_size
        self.dilation = conv.dilation
        self.padding = conv.padding
        self.stride = conv.stride

    def inputs(self, layer_input: Float[torch.Tensor, "batch c_in h w"]) -> Float[torch.Tensor, "batch seq patch"]:
        patches = F.unfold(
            layer_input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride
        )
        return patches.transpose(1, 2)

    def outputs(self, layer_output: Float[torch.Tensor, "batch c_out h w"]) -> Float[torch.Tensor, "batch seq c_out"]:
        return layer_output.flatten(2).transpose(1, 2)

    def weight(self, layer_weight: Float[torch.Tensor, "c_out c_in kh kw"]) -> Float[torch.Tensor, "c_out patch"]:
        return layer_weight.flatten(1)


def grid_shape(layer_output: Float[torch.Tensor, "batch c h w"]) -> tuple[int, int]:
    return layer_output.shape[-2], layer_output.shape[-1]
