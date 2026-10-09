"""A small InceptionV1-shaped CNN for STL-10 (96x96), ported from the hiccup-ide `olt` package.

- Every conv is followed by BatchNorm, then ReLU.
  - The conv named `*_pre_relu_conv` is therefore the pre-BN value, not the pre-ReLU one.
  - BN is a per-channel affine, so a channel with a negative scale flips the sign of the conv output.
  - Hit polarity on a conv layer is read with that in mind.
- Each ReLU is its own `nn.ReLU` module, since captum's DeepLift only sees modules, not functional ops.
- The stem strides by 2 (96 to 48) and its pool halves again (to 24).
  - `block_a` and `block_b` run at 24x24, `downpool` takes them to 12x12 for `block_c` and `block_d`.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 10


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Padding keeps the spatial size, so branches of different kernel sizes can be concatenated."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class StlInceptionBlock(nn.Module):
    """Four branches on the same input, concatenated as 1x1, 3x3, 5x5, pool.

    Arguments follow torchvision's GoogLeNet `Inception`:
    (in, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj).
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        self.branch_1x1 = conv_bn_relu(in_channels, ch1x1, 1)
        self.branch_3x3 = nn.Sequential(conv_bn_relu(in_channels, ch3x3_reduce, 1), conv_bn_relu(ch3x3_reduce, ch3x3, 3))
        self.branch_5x5 = nn.Sequential(conv_bn_relu(in_channels, ch5x5_reduce, 1), conv_bn_relu(ch5x5_reduce, ch5x5, 5))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), conv_bn_relu(in_channels, pool_proj, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch_1x1, self.branch_3x3, self.branch_5x5, self.branch_pool)
        return torch.cat([branch(x) for branch in branches], dim=1)


class StlInception(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block_a = StlInceptionBlock(64, 32, 48, 64, 8, 16, 16)
        self.block_b = StlInceptionBlock(128, 64, 64, 96, 16, 32, 32)
        self.downpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block_c = StlInceptionBlock(224, 96, 64, 128, 16, 32, 32)
        self.block_d = StlInceptionBlock(288, 112, 72, 144, 16, 48, 48)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.block_d.out_channels, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_b(self.block_a(self.stem(x)))
        x = self.block_d(self.block_c(self.downpool(x)))
        return self.head(x)
