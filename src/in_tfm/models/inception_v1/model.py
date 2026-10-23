"""The original InceptionV1 (Lucid's `inception5h`), with lucent's layer names and weights.

- `mixed4e:55` in Lucid is channel 55 of the concatenated `mixed4e`, which is 832 wide.
  - Channels 0 to 255 are the 1x1 branch, so it is channel 55 of `mixed4e_1x1_pre_relu_conv`, before its ReLU.
- The weights are fetched from GitHub by `torch.hub` on first use.
- Inputs are RGB in `[0, 1]`, mapped to `x * 255 - 117` as Lucid does, see `INPUT_MEAN` and `INPUT_STD`.

Why not lucent's own forward:

- `deeplift_attribution` finds the nonlinearities as `nn.ReLU`, `nn.MaxPool2d` and `nn.LocalResponseNorm` modules.
  - Lucent uses its own layer classes for the first two, and calls `F.local_response_norm` as a function.
  - The local response norm is far from the identity on this network's input scale, so leaving it out breaks completeness.
- Lucent's default `RedirectedReLU` has a deliberately wrong backward pass, made for feature visualisation.
  - Attribution needs the real one, so it is never used here.
- `forward` here is the same computation, checked against lucent's in `tests/test_inception_v1.py`.
"""

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SIZE = 224
"""The final average pool is 7x7, which only fits a 224 input."""

INPUT_MEAN = 117 / 255
INPUT_STD = 1 / 255
"""`(x - mean) / std` is `x * 255 - 117`, so the usual normalisation transform reproduces Lucid's input."""

BLOCKS_BEFORE_DOWNSAMPLE = {"mixed3b": "maxpool4", "mixed4e": "maxpool10"}
"""The block after which a stride 2 pool halves the grid, and that pool's name."""

BLOCK_NAMES = ("mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b")

NEGATIVE_INFINITY = float("-inf")


def local_response_norm() -> nn.LocalResponseNorm:
    return nn.LocalResponseNorm(size=9, alpha=9.99999974738e-05, beta=0.5, k=1)


def attributable_layers(lucent_model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Lucent's layers, with its relu and pool classes swapped for the torch modules. Neither has parameters."""
    from lucent.modelzoo.inceptionv1.helper_layers import MaxPool2dLayer, ReluLayer

    def swapped(name: str, layer: nn.Module) -> nn.Module:
        if isinstance(layer, ReluLayer):
            return nn.ReLU()
        if isinstance(layer, MaxPool2dLayer):
            return nn.MaxPool2d(kernel_size=3, stride=1 if name.endswith("_pool") else 2)
        return layer

    return [(name, swapped(name, layer)) for name, layer in lucent_model.named_children()]


class AttributableInceptionV1(nn.Module):
    """Holds lucent's layers under lucent's names, so `mixed4e_1x1_pre_relu_conv` is an attribute."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        for name, layer in attributable_layers(load_lucent_model(pretrained)):
            self.add_module(name, layer)
        self.lrn0 = local_response_norm()
        self.lrn1 = local_response_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for name in BLOCK_NAMES:
            x = self.block(name, x)
            if name in BLOCKS_BEFORE_DOWNSAMPLE:
                x = self.downsample(BLOCKS_BEFORE_DOWNSAMPLE[name], x)
        pooled = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0, count_include_pad=False)
        logits = self.softmax2_pre_activation_matmul(pooled.reshape(-1, 1024))
        return self.softmax2(logits)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d0(self.conv2d0_pre_relu_conv(F.pad(x, (2, 3, 2, 3))))
        x = self.lrn0(self.downsample("maxpool0", x))
        x = self.conv2d1(self.conv2d1_pre_relu_conv(x))
        x = self.conv2d2(self.conv2d2_pre_relu_conv(F.pad(x, (1, 1, 1, 1))))
        return self.downsample("maxpool1", self.lrn1(x))

    def downsample(self, pool_name: str, x: torch.Tensor) -> torch.Tensor:
        """Pads right and bottom only, with -inf so padding never wins a window, as TensorFlow's SAME does."""
        return getattr(self, pool_name)(F.pad(x, (0, 1, 0, 1), value=NEGATIVE_INFINITY))

    def conv_relu(self, block: str, branch: str, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"{block}_{branch}")(getattr(self, f"{block}_{branch}_pre_relu_conv")(x))

    def block(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Four branches on the same input, concatenated as 1x1, 3x3, 5x5, pool projection."""
        one = self.conv_relu(name, "1x1", x)
        three = self.conv_relu(name, "3x3", F.pad(self.conv_relu(name, "3x3_bottleneck", x), (1, 1, 1, 1)))
        five = self.conv_relu(name, "5x5", F.pad(self.conv_relu(name, "5x5_bottleneck", x), (2, 2, 2, 2)))
        pooled = getattr(self, f"{name}_pool")(F.pad(x, (1, 1, 1, 1), value=NEGATIVE_INFINITY))
        projected = self.conv_relu(name, "pool_reduce", pooled)
        return getattr(self, name)((one, three, five, projected), 1)


def load_lucent_model(pretrained: bool) -> nn.Module:
    try:
        from lucent.modelzoo import inceptionv1
    except ImportError as error:
        raise ImportError("InceptionV1 needs lucent, install it with `uv sync --extra inception`") from error
    return inceptionv1(pretrained=pretrained, redirected_ReLU=False)


def load_inception_v1() -> AttributableInceptionV1:
    return AttributableInceptionV1().eval()
