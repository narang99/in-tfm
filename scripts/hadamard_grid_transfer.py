#!/usr/bin/env python
"""Turn one image's whole Hadamard grid into another's: start from a car, fit every cell to a cat's.

- The cat and the car are the highest-activation hits of the named cat trees and car trees.
- The loss is the MSE between the two grids over every cell and every patch dim, plus TV on the pixels.
  - The random affine warps the image and the cat's grid together, as the map pipeline warps its target.
  - The last steps of each block skip the affine.
- At every checkpoint, the cells where the cat fires are placed among the hits, by the forest's distance.
  - Agreement is the share of those cells whose nearest tree matches the cat cell's nearest tree.

    python scripts/hadamard_grid_transfer.py --hit-space hit_space.npz --image-dir data/inception_pool \
        --cat-trees 1,2,23,27 --car-trees 0,3,4,15,18 --out reports/hadamard_grid_transfer
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pydantic import BaseModel

from feature_vis_walk import display_image, to_image, to_pixels
from in_tfm.device import default_device
from in_tfm.hit_space import HitGraph, HitSpace
from in_tfm.sparse_feature_vis import (
    BACKGROUND_GREY,
    HadamardReader,
    load_lucent_inceptionv1,
    seed_thetas,
    total_variation_per_sample,
    warp,
)

HADAMARD_CONV = "mixed4e_1x1_pre_relu_conv"
CHANNEL = 55
TILE_PX = 224


class TransferConfig(BaseModel):
    blocks: int = 12
    steps_per_block: int = 64
    fine_tune_steps: int = 16
    lr: float = 0.01
    tv_weight: float = 1e5
    l1_weight: float = 0.0
    seeds: int = 2


class Pick(BaseModel):
    point: int
    tree: int
    sample_id: str
    row: int
    col: int
    activation: float


def top_hit(graph: HitGraph, trees: list[int]) -> Pick:
    points = np.flatnonzero(np.isin(graph.trees.tree, trees))
    hits = graph.first_hit_of[points]
    best = points[np.argmax(graph.space.activation[hits])]
    sample_id, row, col = graph.hit_position(int(best))
    return Pick(point=int(best), tree=int(graph.trees.tree[best]), sample_id=sample_id, row=row, col=col,
                activation=float(graph.space.activation[graph.first_hit_of[best]]))


def fit_block(reader: HadamardReader, images: torch.Tensor, target: torch.Tensor, config: TransferConfig, salt: int) -> torch.Tensor:
    _, _, height, width = images.shape
    images = images.clone().requires_grad_(True)
    rngs = [random.Random(seed * 100_003 + salt) for seed in range(len(images))]
    optimizer = torch.optim.Adam([images], lr=config.lr)
    targets = target.expand(len(images), -1, -1, -1)
    for step in range(1, config.steps_per_block + 1):
        transformed = step <= config.steps_per_block - config.fine_tune_steps
        theta = seed_thetas(rngs, height, width, transformed, images.device)
        grid = reader(warp(images, theta, fill=BACKGROUND_GREY))
        fit = ((grid - warp(targets, theta, fill=0.0)) ** 2).mean(dim=(1, 2, 3))
        l1 = images.abs().mean(dim=(1, 2, 3))
        loss = (fit + config.l1_weight * l1 + config.tv_weight * total_variation_per_sample(images)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
    return images.detach()


class Checkpoint(BaseModel):
    step: int
    grid_mse: float
    activation_map_correlation: float
    """Pearson correlation of the image's activation map with the cat's."""
    cat_peak_activation: float
    """The image's activation at the cat's peak cell."""
    tree_agreement: float
    """Share of the cat's firing cells whose nearest tree matches the cat cell's."""
    trees_at_firing_cells: dict[str, int]
    mean_pixel: float


def grid_of(reader: HadamardReader, pixels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reader(pixels)


def nearest_trees(graph: HitGraph, grid: torch.Tensor, cells: np.ndarray) -> np.ndarray:
    """`grid` is one image's (dim, rows, cols)."""
    vectors = grid[:, cells[:, 0], cells[:, 1]].T.cpu().numpy()
    nearest, _ = graph.nearest(vectors)
    return graph.trees.tree[nearest]


def checkpoint(reader: HadamardReader, graph: HitGraph, images: torch.Tensor, target: torch.Tensor, firing: np.ndarray,
               cat_trees_at_firing: np.ndarray, cat_peak: Pick, step: int) -> list[Checkpoint]:
    rows = []
    cat_map = reader.activation(target)[0].flatten().cpu().numpy()
    for seed in range(len(images)):
        grid = grid_of(reader, images[seed : seed + 1])
        activation = reader.activation(grid)[0]
        trees = nearest_trees(graph, grid[0], firing)
        rows.append(Checkpoint(
            step=step, grid_mse=float(((grid - target) ** 2).mean()),
            activation_map_correlation=float(np.corrcoef(activation.flatten().cpu().numpy(), cat_map)[0, 1]),
            cat_peak_activation=float(activation[cat_peak.row, cat_peak.col]),
            tree_agreement=float((trees == cat_trees_at_firing).mean()),
            trees_at_firing_cells={str(t): int(n) for t, n in zip(*np.unique(trees, return_counts=True))},
            mean_pixel=float(images[seed].mean()),
        ))
    return rows


def activation_tile(reader: HadamardReader, pixels: torch.Tensor, scale_to: float) -> Image.Image:
    activation = reader.activation(grid_of(reader, pixels))[0].clamp(min=0).cpu().numpy()
    grey = (np.clip(activation / scale_to, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(grey, mode="L").resize((TILE_PX, TILE_PX), Image.NEAREST).convert("RGB")


def labelled(tile: Image.Image, text: str) -> Image.Image:
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, tile.width, 16], fill="black")
    draw.text((3, 2), text, fill="yellow")
    return tile


def sheet(reader: HadamardReader, cat: torch.Tensor, car: torch.Tensor, frames: list[torch.Tensor], steps: list[int]) -> Image.Image:
    """Row 1: the cat, the car, then seed 0 at each checkpoint. Row 2: their activation maps, on the cat's scale."""
    scale_to = float(reader.activation(grid_of(reader, cat)).max())
    columns = [("cat target", cat), ("car start", car)] + [(f"step {s}", f[0:1]) for s, f in zip(steps, frames)]
    out = Image.new("RGB", (TILE_PX * len(columns), TILE_PX * 2), "black")
    for i, (name, pixels) in enumerate(columns):
        out.paste(labelled(to_image(pixels[0]), name), (i * TILE_PX, 0))
        out.paste(labelled(activation_tile(reader, pixels, scale_to), "channel 55 map"), (i * TILE_PX, TILE_PX))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-space", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--cat-trees", required=True)
    parser.add_argument("--car-trees", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    for field, info in TransferConfig.model_fields.items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=info.annotation, default=info.default)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = TransferConfig(**{field: getattr(args, field) for field in TransferConfig.model_fields})
    graph = HitGraph(HitSpace.load(args.hit_space))
    reader = HadamardReader(load_lucent_inceptionv1(device), HADAMARD_CONV, CHANNEL)
    cat_pick = top_hit(graph, [int(t) for t in args.cat_trees.split(",")])
    car_pick = top_hit(graph, [int(t) for t in args.car_trees.split(",")])
    print(f"cat: {cat_pick}\ncar: {car_pick}", flush=True)

    cat = to_pixels(display_image(args.image_dir / cat_pick.sample_id), device)
    car = to_pixels(display_image(args.image_dir / car_pick.sample_id), device)
    target = grid_of(reader, cat)
    cat_activation = reader.activation(target)[0].cpu().numpy()
    firing = np.argwhere(cat_activation >= graph.space.threshold)
    cat_trees_at_firing = nearest_trees(graph, target[0], firing)
    print(f"cat fires at {len(firing)} cells, nearest trees {dict(zip(*np.unique(cat_trees_at_firing, return_counts=True)))}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    images = car.repeat(config.seeds, 1, 1, 1)
    checkpoints = [checkpoint(reader, graph, images, target, firing, cat_trees_at_firing, cat_pick, 0)]
    frames, steps = [], []
    for block in range(1, config.blocks + 1):
        images = fit_block(reader, images, target, config, salt=block)
        step = block * config.steps_per_block
        checkpoints.append(checkpoint(reader, graph, images, target, firing, cat_trees_at_firing, cat_pick, step))
        frames.append(images)
        steps.append(step)
        to_image(images[0]).save(frames_dir / f"step_{step:04d}.png")
        first = checkpoints[-1][0]
        print(f"step {step}: grid mse {first.grid_mse:.2f}, map corr {first.activation_map_correlation:.2f}, "
              f"tree agreement {first.tree_agreement:.2f}, mean pixel {first.mean_pixel:.3f}", flush=True)

    to_image(cat[0]).save(args.out / "cat_target.png")
    to_image(car[0]).save(args.out / "car_start.png")
    shown = sorted(set(np.linspace(0, len(frames) - 1, 6).round().astype(int)))
    sheet(reader, cat, car, [frames[i] for i in shown], [steps[i] for i in shown]).save(args.out / "transfer.png")
    record = {
        "config": config.model_dump(), "cat": cat_pick.model_dump(), "car": car_pick.model_dump(),
        "firing_cells": firing.tolist(), "cat_trees_at_firing": cat_trees_at_firing.tolist(),
        "checkpoints": [[c.model_dump() for c in row] for row in checkpoints],
    }
    (args.out / "transfer.json").write_text(json.dumps(record, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
