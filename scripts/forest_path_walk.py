#!/usr/bin/env python
"""Walk an image along the density-peak forest's own path, one hit at a time.

- From a start image, take the nearest hit to its Hadamard vector at one cell.
- Take the geodesic over the kNN graph from that hit to the closest point of the named cores.
- For each next point on the path, fit the current image so the cell's Hadamard vector matches that point's.
  - Each fit starts from the previous fit's image, so the image moves one graph step at a time.
- The baseline fits the start straight to the path's last point, for the same number of steps.
  - It is checked at the same step counts, so the two can be compared point by point.

    python scripts/forest_path_walk.py --hit-space hit_space.npz --image-dir data/inception_pool \
        --target cats=1,2,23,27 --target snouts=20,24,31,36,39 --out reports/forest_path_walk
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pydantic import BaseModel

from feature_vis_walk import (
    Context,
    StartImage,
    background_starts,
    hit_crop,
    labelled,
    to_image,
    vectors_at,
)
from in_tfm.device import default_device
from in_tfm.hit_space import HitGraph, HitSpace
from in_tfm.sparse_feature_vis import (
    BACKGROUND_GREY,
    HadamardObjective,
    HadamardReader,
    PeakHadamard,
    load_lucent_inceptionv1,
    seed_thetas,
    total_variation_per_sample,
    warp,
)

HADAMARD_CONV = "mixed4e_1x1_pre_relu_conv"
CHANNEL = 55
CROP_PX = 96


class StepConfig(BaseModel):
    steps_per_point: int = 64
    fine_tune_steps: int = 16
    """The last steps of each fit skip the affine, so the fit is judged on the image itself."""
    lr: float = 0.01
    tv_weight: float = 1e3
    l1_weight: float = 0.0
    """L1 on the pixels, as in the sparse runs. It pushes every pixel the target does not need to black."""
    seeds: int = 2


Cell = tuple[int, int]


def fit_to(ctx: Context, images: torch.Tensor, cells: list[Cell], target: np.ndarray, config: StepConfig, salt: int) -> torch.Tensor:
    """Moves `images` so every cell's Hadamard vector matches `target`, with the sparse runs' affine and its bilinear read.

    - One forward pass serves all cells: each is read where the affine moved it, and their MSEs are averaged."""
    target_tensor = torch.from_numpy(target).float().to(ctx.device)
    objectives = [
        HadamardObjective(ctx.reader, PeakHadamard(row=row, col=col, activation=0.0, target=target_tensor), 1.0) for row, col in cells
    ]
    _, _, height, width = images.shape
    images = images.clone().requires_grad_(True)
    rngs = [random.Random(seed * 100_003 + salt) for seed in range(len(images))]
    optimizer = torch.optim.Adam([images], lr=config.lr)
    for step in range(1, config.steps_per_point + 1):
        transformed = step <= config.steps_per_point - config.fine_tune_steps
        theta = seed_thetas(rngs, height, width, transformed, ctx.device)
        grid = ctx.reader(warp(images, theta, fill=BACKGROUND_GREY))
        fit = torch.stack([((o.sample_at_peak(grid, theta) - target_tensor) ** 2).mean(dim=1) for o in objectives]).mean(dim=0)
        l1 = images.abs().mean(dim=(1, 2, 3))
        loss = (fit + config.l1_weight * l1 + config.tv_weight * total_variation_per_sample(images)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
    return images.detach()


class Checkpoint(BaseModel):
    """One seed's state after one fit. Cell measures are averaged over the cells unless named per cell."""

    target_point: int
    target_tree: int
    hadamard_mse: float
    cosine: float
    activation: float
    nearest: int
    """The first cell's nearest point."""
    nearest_tree: int
    nearest_per_cell: list[int]
    reached_target: bool
    """Every cell's nearest point is the target."""
    cells_reached: float
    hops_to_target: int
    """The median over cells."""
    distance_over_tau: float
    pixel_step_rms: float
    """RMS pixel change from the previous checkpoint."""
    exact_zero_share: float
    """Pixels black in every channel."""
    mean_pixel: float


def cell_vectors(ctx: Context, images: torch.Tensor, cells: list[Cell]) -> np.ndarray:
    """(seeds, cells, dim)."""
    return np.stack([vectors_at(ctx.reader, images, row, col) for row, col in cells], axis=1)


def checkpoint(ctx: Context, images: torch.Tensor, previous: torch.Tensor, cells: list[Cell], point: int, target: np.ndarray) -> list[Checkpoint]:
    vectors = cell_vectors(ctx, images, cells)
    seeds, n_cells, _ = vectors.shape
    nearest, distance = ctx.graph.nearest(vectors.reshape(seeds * n_cells, -1))
    nearest, distance = nearest.reshape(seeds, n_cells), distance.reshape(seeds, n_cells)
    hops = ctx.graph.hops_between(nearest.ravel(), np.full(nearest.size, point)).reshape(seeds, n_cells)
    rms = ((images - previous) ** 2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
    rows = []
    for seed in range(seeds):
        cosines = vectors[seed] @ target / (np.linalg.norm(vectors[seed], axis=1) * np.linalg.norm(target) + 1e-12)
        rows.append(Checkpoint(
            target_point=point, target_tree=int(ctx.graph.trees.tree[point]),
            hadamard_mse=float(((vectors[seed] - target) ** 2).mean()), cosine=float(cosines.mean()),
            activation=float(vectors[seed].sum(axis=1).mean() + float(ctx.reader.bias)),
            nearest=int(nearest[seed, 0]), nearest_tree=int(ctx.graph.trees.tree[nearest[seed, 0]]),
            nearest_per_cell=nearest[seed].tolist(), reached_target=bool((nearest[seed] == point).all()),
            cells_reached=float((nearest[seed] == point).mean()), hops_to_target=int(np.median(hops[seed])),
            distance_over_tau=float(distance[seed].mean() / ctx.graph.tau), pixel_step_rms=float(rms[seed]),
            exact_zero_share=float((images[seed].amax(0) == 0).float().mean()), mean_pixel=float(images[seed].mean()),
        ))
    return rows


def point_vector(ctx: Context, point: int) -> np.ndarray:
    return ctx.graph.space.hadamard[ctx.graph.first_hit_of[point]].astype(np.float32)


class Walk(BaseModel):
    start: str
    target_name: str
    target_trees: list[int]
    cells: list[Cell]
    start_point: int
    start_distance_over_tau: float
    path: list[int]
    path_trees: list[int]
    path_length: float
    stepwise: list[list[Checkpoint]]
    """Per path point after the start, one checkpoint per seed."""
    direct: list[list[Checkpoint]]
    """Per block of the same length, one checkpoint per seed, always aimed at the path's last point."""


def walk(ctx: Context, start: StartImage, cells: list[Cell], target_name: str, target_trees: list[int], config: StepConfig) -> tuple[Walk, list[torch.Tensor], list[torch.Tensor]]:
    """The path starts at the nearest hit to the mean of the cells' start vectors."""
    images = start.pixels.repeat(config.seeds, 1, 1, 1)
    start_vector = cell_vectors(ctx, start.pixels, cells)[0].mean(axis=0, keepdims=True)
    start_point, start_distance = ctx.graph.nearest(start_vector)
    path, length = ctx.graph.shortest_path_to(int(start_point[0]), ctx.graph.core_points(target_trees))
    print(f"{start.name} -> {target_name} on cells {cells}: start tree {ctx.graph.trees.tree[path[0]]}, path of {len(path)} points "
          f"through trees {list(dict.fromkeys(int(ctx.graph.trees.tree[p]) for p in path))}", flush=True)
    stepwise, stepwise_images = [], []
    current = images
    for index, point in enumerate(path[1:], start=1):
        moved = fit_to(ctx, current, cells, point_vector(ctx, point), config, salt=index)
        stepwise.append(checkpoint(ctx, moved, current, cells, point, point_vector(ctx, point)))
        stepwise_images.append(moved)
        current = moved
    direct, direct_images = [], []
    current = images
    final_point = path[-1]
    for index in range(1, len(path)):
        moved = fit_to(ctx, current, cells, point_vector(ctx, final_point), config, salt=index)
        direct.append(checkpoint(ctx, moved, current, cells, final_point, point_vector(ctx, final_point)))
        direct_images.append(moved)
        current = moved
    record = Walk(
        start=start.name, target_name=target_name, target_trees=target_trees, cells=cells,
        start_point=int(start_point[0]), start_distance_over_tau=float(start_distance[0] / ctx.graph.tau),
        path=path, path_trees=[int(ctx.graph.trees.tree[p]) for p in path], path_length=length,
        stepwise=stepwise, direct=direct,
    )
    return record, stepwise_images, direct_images


def cells_centre_px(cells: list[Cell]) -> tuple[float, float]:
    """(x, y) in pixels of the middle of the cells' block."""
    return float(np.mean([c * 16 + 8 for _, c in cells])), float(np.mean([r * 16 + 8 for r, _ in cells]))


def crop_around_cells(image: Image.Image, cells: list[Cell]) -> Image.Image:
    x, y = cells_centre_px(cells)
    half = CROP_PX // 2
    return image.crop((int(x - half), int(y - half), int(x + half), int(y + half)))


def cells_box_px(cells: list[Cell], scale: float) -> list[int]:
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    return [int(min(cols) * 16 * scale), int(min(rows) * 16 * scale), int((max(cols) + 1) * 16 * scale), int((max(rows) + 1) * 16 * scale)]


def step_label(step: Checkpoint, n_cells: int) -> str:
    if n_cells == 1:
        return f"{'hit' if step.reached_target else f'+{step.hops_to_target}h'} t{step.nearest_tree}"
    return f"{round(step.cells_reached * n_cells)}/{n_cells} t{step.nearest_tree}"


def walk_sheet(ctx: Context, start: StartImage, record: Walk, stepwise: list[torch.Tensor], direct: list[torch.Tensor]) -> Image.Image:
    """Row 1: the path's hits. Row 2: the stepwise image after reaching each. Row 3: the direct fit at the same step count."""
    columns = len(record.path)
    cells = [tuple(c) for c in record.cells]
    sheet = Image.new("RGB", (CROP_PX * columns, CROP_PX * 3), "black")
    start_crop = crop_around_cells(to_image(start.pixels[0]), cells)
    for column, point in enumerate(record.path):
        sheet.paste(labelled(hit_crop(ctx, point), f"t{record.path_trees[column]}").resize((CROP_PX, CROP_PX)), (column * CROP_PX, 0))
    sheet.paste(labelled(start_crop, "start"), (0, CROP_PX))
    sheet.paste(labelled(start_crop, "start"), (0, 2 * CROP_PX))
    for column in range(1, columns):
        step = record.stepwise[column - 1][0]
        tile = crop_around_cells(to_image(stepwise[column - 1][0]), cells)
        sheet.paste(labelled(tile, step_label(step, len(cells))), (column * CROP_PX, CROP_PX))
        jump = record.direct[column - 1][0]
        tile = crop_around_cells(to_image(direct[column - 1][0]), cells)
        sheet.paste(labelled(tile, step_label(jump, len(cells))), (column * CROP_PX, 2 * CROP_PX))
    return sheet


def full_strip(start: StartImage, record: Walk, stepwise: list[torch.Tensor], direct: list[torch.Tensor], size: int = 160) -> Image.Image:
    """Whole images: row 1 stepwise after each point, row 2 the direct fit at the same step count. The cells are boxed."""
    columns = len(record.path)
    sheet = Image.new("RGB", (size * columns, size * 2), "white")
    box = cells_box_px([tuple(c) for c in record.cells], size / start.pixels.shape[-1])
    rows = [[start.pixels[0]] + [images[0] for images in stepwise], [start.pixels[0]] + [images[0] for images in direct]]
    for row, frames in enumerate(rows):
        for column, frame in enumerate(frames):
            tile = to_image(frame).resize((size, size))
            ImageDraw.Draw(tile).rectangle(box, outline="red")
            sheet.paste(tile, (column * size, row * size))
    return sheet


def summarise(record: Walk) -> dict:
    def per_seed(rows: list[list[Checkpoint]], field: str) -> np.ndarray:
        return np.array([[getattr(c, field) for c in step] for step in rows])

    def path_mean(rows: list[list[Checkpoint]], field: str) -> float:
        return float(per_seed(rows, field).mean())

    final_step, final_direct = record.stepwise[-1], record.direct[-1]
    return {
        "start": record.start, "target": record.target_name, "path_points": len(record.path),
        "path_trees": list(dict.fromkeys(record.path_trees)), "start_distance_over_tau": record.start_distance_over_tau,
        "cells": record.cells,
        "stepwise_share_reached": path_mean(record.stepwise, "reached_target"),
        "stepwise_share_cells_reached": path_mean(record.stepwise, "cells_reached"),
        "direct_share_cells_reached": path_mean(record.direct, "cells_reached"),
        "stepwise_median_hops_off": float(np.median(per_seed(record.stepwise, "hops_to_target"))),
        "stepwise_mean_distance_over_tau": path_mean(record.stepwise, "distance_over_tau"),
        "direct_mean_distance_over_tau": path_mean(record.direct, "distance_over_tau"),
        "stepwise_final_mse": float(np.mean([c.hadamard_mse for c in final_step])),
        "direct_final_mse": float(np.mean([c.hadamard_mse for c in final_direct])),
        "stepwise_final_reached": float(np.mean([c.reached_target for c in final_step])),
        "direct_final_reached": float(np.mean([c.reached_target for c in final_direct])),
        "stepwise_total_pixel_rms": float(per_seed(record.stepwise, "pixel_step_rms").sum(axis=0).mean()),
        "direct_total_pixel_rms": float(per_seed(record.direct, "pixel_step_rms").sum(axis=0).mean()),
        "stepwise_final_tree": [c.nearest_tree for c in final_step],
        "direct_final_tree": [c.nearest_tree for c in final_direct],
        "direct_trees_on_the_way": list(dict.fromkeys(c.nearest_tree for step in record.direct for c in step[:1])),
        "stepwise_final_zero_share": float(np.mean([c.exact_zero_share for c in final_step])),
        "direct_final_zero_share": float(np.mean([c.exact_zero_share for c in final_direct])),
        "stepwise_final_mean_pixel": float(np.mean([c.mean_pixel for c in final_step])),
    }


def parse_cells(text: str) -> list[Cell]:
    return [tuple(int(v) for v in cell.split(",")) for cell in text.split(";")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-space", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", required=True, help="name=comma-separated tree ids")
    parser.add_argument("--background-tree", type=int, default=0)
    parser.add_argument("--background-starts", type=int, default=3)
    parser.add_argument("--with-grey", action="store_true")
    parser.add_argument("--only-starts", default="", help="comma-separated start names to keep, for example background_t0_0")
    parser.add_argument("--cells", default="", help="row,col cells fitted together, for example '6,6;6,7;7,6;7,7'. Default: each start's own cell")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    for field, info in StepConfig.model_fields.items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=info.annotation, default=info.default)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = StepConfig(**{field: getattr(args, field) for field in StepConfig.model_fields})
    targets = {name: [int(t) for t in trees.split(",")] for name, trees in (entry.split("=", 1) for entry in args.target)}
    ctx = Context(
        reader=HadamardReader(load_lucent_inceptionv1(device), HADAMARD_CONV, CHANNEL), graph=HitGraph(HitSpace.load(args.hit_space)),
        image_dir=args.image_dir, cat_trees=next(iter(targets.values())), snout_trees=list(targets.values())[-1], device=device,
    )
    starts = background_starts(ctx, args.background_tree, args.background_starts)
    if args.with_grey:
        starts.append(StartImage(name="grey", pixels=torch.full((1, 3, 224, 224), BACKGROUND_GREY, device=device), row=7, col=7))
    if args.only_starts:
        keep = set(args.only_starts.split(","))
        starts = [start for start in starts if start.name in keep]
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for start in starts:
        for name, trees in targets.items():
            cells = parse_cells(args.cells) if args.cells else [(start.row, start.col)]
            record, stepwise, direct = walk(ctx, start, cells, name, trees, config)
            stem = f"{start.name}_to_{name}"
            (args.out / f"{stem}.json").write_text(record.model_dump_json())
            walk_sheet(ctx, start, record, stepwise, direct).save(args.out / f"{stem}.png")
            full_strip(start, record, stepwise, direct).save(args.out / f"{stem}_full.png")
            frames_dir = args.out / stem
            frames_dir.mkdir(exist_ok=True)
            for index, (moved, jumped) in enumerate(zip(stepwise, direct), start=1):
                to_image(moved[0]).save(frames_dir / f"stepwise_{index:02d}.png")
                to_image(jumped[0]).save(frames_dir / f"direct_{index:02d}.png")
            summaries.append(summarise(record))
            print(json.dumps(summaries[-1]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=1))
    (args.out / "run.json").write_text(json.dumps({"config": config.model_dump(), "targets": targets, "tau": ctx.graph.tau}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
