"""A two-conv network for MNIST, small enough that every neuron can be read.

- Both convs are 3x3 with stride 1 and no padding, so the grid shrinks 28 to 26 to 24.
  - There is no pooling, so every op in the network is stride 1.
  - Without padding no position ever sees a made-up border.
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


class MnistCnnChannels(BaseModel):
    conv1: int = 4
    conv2: int = 8


class MnistCnn(nn.Module):
    def __init__(self, channels: MnistCnnChannels = MnistCnnChannels(), num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, channels.conv1, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels.conv1, channels.conv2, kernel_size=3)
        self.relu2 = nn.ReLU()
        grid = IMAGE_SIZE - 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels.conv2 * grid * grid, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
