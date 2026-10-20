"""A two-conv network for MNIST, small enough that every neuron can be read.

- Both convs are 3x3 with stride 2, so the grid goes 28 to 14 to 6.
  - The linear head then reads a 6x6 grid, so it has too few weights to memorise pixel positions.
  - A global average pool instead lost the layout of the strokes, and accuracy fell below 70%.
- Only conv1 is padded.
  - The input is zero on MNIST's background, so conv1's zero padding looks like more background.
  - conv1's output on background is `relu(bias)`, not zero, so padding conv2 would add a fake border.
  - Unpadded, conv2 skips the last row and column of its 14x14 input, which MNIST leaves blank.
- There is no BatchNorm, so a conv's output is the value its ReLU sees, sign included.
- Each ReLU is its own `nn.ReLU` module, since DeepLift finds the ops it replaces as modules.
- The input is raw pixels in [0, 1], not normalised.
  - DeepLift's zero baseline is then a blank black image, the natural "no digit" reference.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel

NUM_CLASSES = 10
IMAGE_SIZE = 28
HEAD_GRID = 6


class MnistCnnChannels(BaseModel):
    conv1: int = 8
    conv2: int = 16


class MnistCnn(nn.Module):
    def __init__(self, channels: MnistCnnChannels = MnistCnnChannels(), num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, channels.conv1, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels.conv1, channels.conv2, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels.conv2 * HEAD_GRID * HEAD_GRID, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
