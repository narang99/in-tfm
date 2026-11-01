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

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel
from torch import nn

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


class ChannelReader:
    """Reads one channel of one layer, and stops the forward pass there, since later layers never touch the loss."""

    def __init__(self, model: nn.Module, layer: str, channel: int) -> None:
        self.model = model
        self.channel = channel
        self.captured: dict[str, torch.Tensor] = {}
        getattr(model, layer).register_forward_hook(self.capture_and_stop)

    def capture_and_stop(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self.captured["output"] = output
        raise StopForward

    def __call__(self, pixels: Pixels) -> ChannelMap:
        try:
            self.model(to_inceptionv1_input(pixels))
        except StopForward:
            pass
        return self.captured["output"][:, self.channel : self.channel + 1]


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


def optimise_seed_batch(
    read_channel: ChannelReader, reference: Pixels, target: ChannelMap, seeds: list[int], config: SparseVisConfig
) -> Pixels:
    """Per-seed losses are summed and Adam is elementwise, so one batch equals separate runs per seed."""
    _, _, height, width = reference.shape
    images = reference.repeat(len(seeds), 1, 1, 1).clone().requires_grad_(True)
    targets = target.expand(len(seeds), -1, -1, -1)
    rngs = [random.Random(seed) for seed in seeds]
    optimizer = torch.optim.Adam([images], lr=config.lr)
    for step in range(1, config.steps + 1):
        transformed = step <= config.steps - config.fine_tune_steps
        theta = seed_thetas(rngs, height, width, transformed, reference.device)
        produced = read_channel(warp(images, theta, fill=BACKGROUND_GREY))
        mse = ((produced - warp(targets, theta, fill=0.0)) ** 2).mean(dim=(1, 2, 3))
        l1 = images.abs().mean(dim=(1, 2, 3))
        loss = (mse + config.l1_weight * l1 + config.tv_weight * total_variation_per_sample(images)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
        if step % 64 == 0 or step == config.steps:
            print(f"  seeds {seeds[0]}-{seeds[-1]} step {step}/{config.steps}  median map mse {mse.median().item():.1f}", flush=True)
    return images.detach()


def optimise_all_seeds(read_channel: ChannelReader, reference: Pixels, target: ChannelMap, config: SparseVisConfig) -> Pixels:
    all_seeds = list(range(config.seeds))
    batches = [all_seeds[start : start + config.seed_batch] for start in range(0, config.seeds, config.seed_batch)]
    return torch.cat([optimise_seed_batch(read_channel, reference, target, batch, config) for batch in batches])


def channel_maps(read_channel: ChannelReader, images: Pixels, batch_size: int) -> ChannelMap:
    with torch.no_grad():
        return torch.cat([read_channel(images[start : start + batch_size]) for start in range(0, len(images), batch_size)])
