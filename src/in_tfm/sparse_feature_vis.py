"""Sparse images that keep one InceptionV1 channel's strong activations, ported from `l1-on-feat-vis`.

- Each seed starts from the photo and optimises its raw pixels against
  `MSE(map, thresholded map) + l1 * mean|x| + tv * TV(x)`.
  - A random affine warps the image and the target map together, so the cells stay aligned.
    - Without it the optimiser finds colour noise that matches the map exactly.
  - The last `fine_tune_steps` skip the affine, so the final fit is on the saved image.
- The seeds are averaged.
  - Each seed fits the map partly through its own edits that suppress the cells around the core.
  - The average cancels those edits and keeps the image structure that all seeds share.
- The model is lucent's own `inceptionv1`, with its `RedirectedReLU`, as in the original.
  - `models/inception_v1` swaps that for the true ReLU, which attribution needs and feature vis does not.
"""

import math
import random
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel
from torch import nn

from .token_layout import ConvLayout

Pixels = Float[torch.Tensor, "batch 3 height width"]
ChannelMap = Float[torch.Tensor, "batch 1 map_height map_width"]
AffineTheta = Float[torch.Tensor, "batch 2 3"]

MAX_SHIFT_PX = 12
SCALES = [1 + (i - 5) / 50.0 for i in range(11)]
ANGLES_DEG = list(range(-10, 11)) + 5 * [0]
"""The extra zeros make an unrotated draw more likely, as in lucent's `standard_transforms`."""

BACKGROUND_GREY = 0.5


class SparseVisConfig(BaseModel):
    threshold: float = 600.0
    threshold_frac: float = 0.0
    """When above 0, the threshold is this fraction of each image's own peak."""
    size: int = 448
    l1_weight: float = 3e5
    tv_weight: float = 1e5
    lr: float = 0.01
    steps: int = 768
    fine_tune_steps: int = 96
    seeds: int = 20
    seed_batch: int = 20
    """Seeds per forward pass. Smaller batches use less memory and give the same result."""
    objective: Literal["map", "hadamard"] = "map"
    """`hadamard` fits only the peak position's Hadamard vector, see `HadamardObjective`."""
    hadamard_conv: str = "mixed4e_1x1_pre_relu_conv"
    """The conv whose patch the Hadamard vector splits. Its channel is the neuron's channel."""
    hadamard_weight: float = 1.0


class StopForward(Exception):
    pass


def to_inceptionv1_input(pixels: Pixels) -> torch.Tensor:
    return pixels * 255 - 117


def total_variation_per_sample(images: Pixels) -> Float[torch.Tensor, " batch"]:
    dx = (images[..., :, 1:] - images[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (images[..., 1:, :] - images[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return dx + dy


def random_affine_theta(rng: random.Random, height: int, width: int) -> list[list[float]]:
    """Output to input mapping in normalised coordinates.

    - The rotation is in pixel space.
      - The normalised x and y axes have different pixel lengths on a non-square image, so they are rescaled around it.
    """
    angle = math.radians(rng.choice(ANGLES_DEG))
    scale = rng.choice(SCALES)
    shift_x = rng.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (width / 2)
    shift_y = rng.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (height / 2)
    aspect = height / width
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    return [[cos, -sin * aspect, shift_x], [sin / aspect, cos, shift_y]]


IDENTITY_THETA = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def warp(tensor: torch.Tensor, theta: AffineTheta, fill: float) -> torch.Tensor:
    grid = F.affine_grid(theta, list(tensor.shape), align_corners=False)
    return F.grid_sample(tensor - fill, grid, align_corners=False) + fill


def load_lucent_inceptionv1(device: torch.device) -> nn.Module:
    from lucent.modelzoo import inceptionv1

    return inceptionv1(pretrained=True).to(device).eval()


def forward_until(model: nn.Module, module: nn.Module, pixels: Pixels, take_input: bool) -> torch.Tensor:
    """The input or output of `module`, with the forward pass stopped there, since later layers never touch the loss.

    - The hook lives only for this call.
      - Two readers on one model would otherwise stop each other's forward pass.
    """
    captured: dict[str, torch.Tensor] = {}

    def capture_input(module: nn.Module, inputs: tuple) -> None:
        captured["tensor"] = inputs[0]
        raise StopForward

    def capture_output(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured["tensor"] = output
        raise StopForward

    handle = module.register_forward_pre_hook(capture_input) if take_input else module.register_forward_hook(capture_output)
    try:
        model(to_inceptionv1_input(pixels))
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured["tensor"]


class ChannelReader:
    def __init__(self, model: nn.Module, layer: str, channel: int) -> None:
        self.model = model
        self.layer = getattr(model, layer)
        self.channel = channel

    def __call__(self, pixels: Pixels) -> ChannelMap:
        return forward_until(self.model, self.layer, pixels, take_input=False)[:, self.channel : self.channel + 1]


HadamardGrid = Float[torch.Tensor, "batch patch map_height map_width"]
HadamardVector = Float[torch.Tensor, " patch"]


class HadamardReader:
    """`patch * weight[channel]` at every output position of one conv.

    - It sums, plus the bias, to the conv's pre-ReLU output, so it splits the activation over the patch.
    - The patch is the conv's own input, unfolded as in `token_layout.ConvLayout`.
    """

    def __init__(self, model: nn.Module, conv_name: str, channel: int) -> None:
        self.model = model
        self.conv: nn.Conv2d = getattr(model, conv_name)
        self.layout = ConvLayout(self.conv)
        self.kernel_row = self.layout.weight(self.conv.weight)[channel]
        self.bias = self.conv.bias[channel].detach() if self.conv.bias is not None else torch.zeros(())

    def __call__(self, pixels: Pixels) -> HadamardGrid:
        conv_input = forward_until(self.model, self.conv, pixels, take_input=True)
        map_height, map_width = self.conv(conv_input[:1, :, :, :]).shape[-2:]
        products = self.layout.inputs(conv_input) * self.kernel_row
        return products.transpose(1, 2).reshape(len(pixels), -1, map_height, map_width)

    def activation(self, grid: HadamardGrid) -> Float[torch.Tensor, "batch map_height map_width"]:
        return grid.sum(1) + self.bias


class PeakHadamard(BaseModel, arbitrary_types_allowed=True):
    row: int
    col: int
    activation: float
    target: HadamardVector


def peak_hadamard(reader: HadamardReader, reference: Pixels) -> PeakHadamard:
    with torch.no_grad():
        grid = reader(reference)
        activation = reader.activation(grid)[0]
    row, col = divmod(activation.argmax().item(), activation.shape[1])
    return PeakHadamard(row=row, col=col, activation=activation[row, col].item(), target=grid[0, :, row, col])


class SparseTarget(BaseModel, arbitrary_types_allowed=True):
    activation_map: ChannelMap
    target: ChannelMap
    threshold: float

    @property
    def cell_count(self) -> int:
        return int((self.target > 0).sum())


def build_target(read_channel: ChannelReader, reference: Pixels, config: SparseVisConfig) -> SparseTarget:
    with torch.no_grad():
        activation_map = read_channel(reference)
    peak = activation_map.max().item()
    threshold = config.threshold_frac * peak if config.threshold_frac > 0 else config.threshold
    target = torch.where(activation_map >= threshold, activation_map, 0.0)
    return SparseTarget(activation_map=activation_map, target=target, threshold=threshold)


def seed_thetas(rngs: list[random.Random], height: int, width: int, transformed: bool, device: torch.device) -> AffineTheta:
    rows = [random_affine_theta(rng, height, width) if transformed else IDENTITY_THETA for rng in rngs]
    return torch.tensor(rows, dtype=torch.float32, device=device)


PerSeedLoss = Float[torch.Tensor, " batch"]


class FitObjective(Protocol):
    """How far a warped batch is from the target, with the target warped by the same `theta`."""

    def __call__(self, warped_images: Pixels, theta: AffineTheta) -> PerSeedLoss: ...


class MapObjective:
    def __init__(self, read_channel: ChannelReader, target: ChannelMap) -> None:
        self.read_channel = read_channel
        self.target = target

    def __call__(self, warped_images: Pixels, theta: AffineTheta) -> PerSeedLoss:
        targets = warp(self.target.expand(len(warped_images), -1, -1, -1), theta, fill=0.0)
        return ((self.read_channel(warped_images) - targets) ** 2).mean(dim=(1, 2, 3))


def cell_centre(row: int, col: int, map_height: int, map_width: int) -> tuple[float, float]:
    """Normalised (x, y) of a map cell's centre, in `align_corners=False` coordinates."""
    return (2 * col + 1) / map_width - 1, (2 * row + 1) / map_height - 1


def where_theta_moves(point: tuple[float, float], theta: AffineTheta) -> Float[torch.Tensor, "batch 2"]:
    """Where `point` of the unwarped image lands in each warped image.

    - `theta` maps a warped position `p` to the source position `A p + t`.
      - So the source `point` lands at `A^-1 (point - t)`.
    """
    linear, shift = theta[:, :, :2], theta[:, :, 2]
    source = torch.tensor(point, dtype=theta.dtype, device=theta.device).expand_as(shift)
    return torch.linalg.solve(linear, source - shift)


class HadamardObjective:
    """MSE between one position's Hadamard vector and the reference's, at the reference peak.

    - The affine moves the peak off the cell grid, so the grid is sampled bilinearly where the peak landed.
    """

    def __init__(self, reader: HadamardReader, peak: PeakHadamard, weight: float) -> None:
        self.reader = reader
        self.peak = peak
        self.weight = weight

    def sample_at_peak(self, grid: HadamardGrid, theta: AffineTheta) -> Float[torch.Tensor, "batch patch"]:
        centre = cell_centre(self.peak.row, self.peak.col, grid.shape[-2], grid.shape[-1])
        landed = where_theta_moves(centre, theta)
        sampled = F.grid_sample(grid, landed[:, None, None, :], align_corners=False)
        return sampled[:, :, 0, 0]

    def __call__(self, warped_images: Pixels, theta: AffineTheta) -> PerSeedLoss:
        produced = self.sample_at_peak(self.reader(warped_images), theta)
        return self.weight * ((produced - self.peak.target) ** 2).mean(dim=1)


def optimise_seed_batch(objective: FitObjective, reference: Pixels, seeds: list[int], config: SparseVisConfig) -> Pixels:
    """Per-seed losses are summed and Adam is elementwise, so one batch equals separate runs per seed."""
    _, _, height, width = reference.shape
    images = reference.repeat(len(seeds), 1, 1, 1).clone().requires_grad_(True)
    rngs = [random.Random(seed) for seed in seeds]
    optimizer = torch.optim.Adam([images], lr=config.lr)
    for step in range(1, config.steps + 1):
        transformed = step <= config.steps - config.fine_tune_steps
        theta = seed_thetas(rngs, height, width, transformed, reference.device)
        fit = objective(warp(images, theta, fill=BACKGROUND_GREY), theta)
        l1 = images.abs().mean(dim=(1, 2, 3))
        loss = (fit + config.l1_weight * l1 + config.tv_weight * total_variation_per_sample(images)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
        if step % 64 == 0 or step == config.steps:
            print(f"  seeds {seeds[0]}-{seeds[-1]} step {step}/{config.steps}  median fit {fit.median().item():.1f}", flush=True)
    return images.detach()


def optimise_all_seeds(objective: FitObjective, reference: Pixels, config: SparseVisConfig) -> Pixels:
    all_seeds = list(range(config.seeds))
    batches = [all_seeds[start : start + config.seed_batch] for start in range(0, config.seeds, config.seed_batch)]
    return torch.cat([optimise_seed_batch(objective, reference, batch, config) for batch in batches])


def channel_maps(read_channel: ChannelReader, images: Pixels, batch_size: int) -> ChannelMap:
    with torch.no_grad():
        return torch.cat([read_channel(images[start : start + batch_size]) for start in range(0, len(images), batch_size)])
