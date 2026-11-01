#!/usr/bin/env python
"""Does feature visualisation walk the paths that a neuron's own hits lay down?

- The hits and their density-peak forest come from `scripts/build_hit_space.py`.
- Experiment 1, `place`: put a given Hadamard vector among the hits.
  - Its nearest hit, the forest's parent links up to that hit's root, and the geodesic to the named cores.
- Experiment 2, `climb`: maximise the neuron at one cell from a start image, and track that cell's Hadamard vector.
  - Each snapshot is placed among the hits, as in experiment 1.
  - Two straight lines between the same two ends are the baselines.
    - `pixel_lerp` blends the start and final images.
    - `hadamard_lerp` blends the start and final Hadamard vectors.
  - Walking a worn path means staying closer to the hits than the baselines, and moving between graph neighbours.

    python scripts/feature_vis_walk.py --hit-space reports/hit_space/hit_space.npz --image-dir data/inception_pool \
        --cat-image cat.jpg --cat-trees 1,7 --snout-trees 8,20 --out reports/feature_vis_walk
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pydantic import BaseModel

from in_tfm.device import default_device
from in_tfm.hit_space import HitGraph, HitSpace
from in_tfm.sparse_feature_vis import (
    BACKGROUND_GREY,
    HadamardReader,
    load_lucent_inceptionv1,
    peak_hadamard,
    seed_thetas,
    total_variation_per_sample,
    warp,
    where_theta_moves,
    cell_centre,
)

HADAMARD_CONV = "mixed4e_1x1_pre_relu_conv"
CHANNEL = 55
CELL_PX = 16
CROP_PX = 96
STRIP_FRAMES = 12


def display_image(path: Path) -> Image.Image:
    """Resized to 256 and centre-cropped to 224, as the hits were."""
    image = Image.open(path).convert("RGB")
    scale = 256 / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    return image.crop((left, top, left + 224, top + 224))


def short_side_image(path: Path, short_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = short_side / min(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def to_pixels(image: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


def to_image(pixels: torch.Tensor) -> Image.Image:
    array = pixels.detach().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return Image.fromarray((array * 255).round().astype(np.uint8))


def crop_around_cell(image: Image.Image, row: int, col: int) -> Image.Image:
    centre_x, centre_y = col * CELL_PX + CELL_PX // 2, row * CELL_PX + CELL_PX // 2
    half = CROP_PX // 2
    return image.crop((centre_x - half, centre_y - half, centre_x + half, centre_y + half))


def vectors_at(reader: HadamardReader, pixels: torch.Tensor, row: int, col: int, batch: int = 32) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([reader(pixels[s : s + batch])[:, :, row, col] for s in range(0, len(pixels), batch)]).cpu().numpy()


class Context(BaseModel, arbitrary_types_allowed=True):
    reader: HadamardReader
    graph: HitGraph
    image_dir: Path
    cat_trees: list[int]
    snout_trees: list[int]
    device: torch.device


def hit_crop(ctx: Context, point: int) -> Image.Image:
    sample_id, row, col = ctx.graph.hit_position(point)
    return crop_around_cell(display_image(ctx.image_dir / sample_id), row, col)


def labelled(tile: Image.Image, text: str) -> Image.Image:
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, tile.width, 22], fill="black")
    draw.text((2, 1), text, fill="yellow")
    return tile


def strip(tiles: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (CROP_PX * len(tiles), CROP_PX), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((CROP_PX, CROP_PX)), (i * CROP_PX, 0))
    return sheet


def point_label(ctx: Context, point: int) -> str:
    return f"t{ctx.graph.trees.tree[point]} d{ctx.graph.trees.depth[point]}"


def path_strip(ctx: Context, query_tile: Image.Image, path: list[int]) -> Image.Image:
    shown = path if len(path) <= 15 else [path[i] for i in np.linspace(0, len(path) - 1, 15).round().astype(int)]
    return strip([labelled(query_tile, "query")] + [labelled(hit_crop(ctx, p), point_label(ctx, p)) for p in shown])


# ---------- experiment 1: place a given vector ----------


class PlacedQuery(BaseModel):
    name: str
    peak_row: int
    peak_col: int
    activation: float
    nearest: int
    nearest_tree: int
    nearest_distance: float
    distance_over_tau: float
    nearest_distance_percentile: float
    """Where the query's nearest distance falls among the hits' own nearest-neighbour distances."""
    ascent: list[int]
    ascent_trees: list[int]
    snout_path: list[int]
    snout_path_length: float
    snout_path_trees: list[int]
    cat_path: list[int]
    cat_path_length: float


def place_query(ctx: Context, name: str, pixels: torch.Tensor, out_dir: Path) -> PlacedQuery:
    peak = peak_hadamard(ctx.reader, pixels)
    vector = peak.target.cpu().numpy()[None]
    nearest, distance = ctx.graph.nearest(vector)
    point = int(nearest[0])
    own_nearest = ctx.graph.forest.neighbour_distance[:, 0]
    ascent = ctx.graph.ascent(point)
    snout_path, snout_length = ctx.graph.shortest_path_to(point, ctx.graph.core_points(ctx.snout_trees))
    cat_path, cat_length = ctx.graph.shortest_path_to(point, ctx.graph.core_points(ctx.cat_trees))
    query_tile = crop_around_cell(to_image(pixels[0]), peak.row, peak.col)
    for label, path in [("ascent", ascent), ("to_snout", snout_path), ("to_cat", cat_path)]:
        path_strip(ctx, query_tile, path).save(out_dir / f"{name}_{label}.png")
    return PlacedQuery(
        name=name, peak_row=peak.row, peak_col=peak.col, activation=peak.activation, nearest=point,
        nearest_tree=int(ctx.graph.trees.tree[point]), nearest_distance=float(distance[0]),
        distance_over_tau=float(distance[0] / ctx.graph.tau),
        nearest_distance_percentile=float((own_nearest < distance[0]).mean() * 100),
        ascent=ascent, ascent_trees=[int(ctx.graph.trees.tree[p]) for p in ascent],
        snout_path=snout_path, snout_path_length=snout_length,
        snout_path_trees=[int(ctx.graph.trees.tree[p]) for p in snout_path],
        cat_path=cat_path, cat_path_length=cat_length,
    )


# ---------- experiment 2: climb from a start image ----------


class ClimbConfig(BaseModel):
    steps: int = 512
    lr: float = 0.01
    tv_weight: float = 1e4
    seeds: int = 4
    snapshot_every: int = 4


def climb(ctx: Context, start: torch.Tensor, row: int, col: int, config: ClimbConfig) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Maximises the neuron's pre-ReLU output at (row, col), under the same random affine as the sparse runs.

    - The affine moves the cell, so the activation is read bilinearly where the cell landed.
    - Snapshots are taken without the affine, at the cell itself."""
    import random

    _, _, height, width = start.shape
    images = start.repeat(config.seeds, 1, 1, 1).clone().requires_grad_(True)
    rngs = [random.Random(seed) for seed in range(config.seeds)]
    optimizer = torch.optim.Adam([images], lr=config.lr)
    snapshots = [images.detach().clone()]
    for step in range(1, config.steps + 1):
        theta = seed_thetas(rngs, height, width, True, start.device)
        grid = ctx.reader(warp(images, theta, fill=BACKGROUND_GREY))
        activation_map = ctx.reader.activation(grid)[:, None]
        landed = where_theta_moves(cell_centre(row, col, grid.shape[-2], grid.shape[-1]), theta)
        activation = F.grid_sample(activation_map, landed[:, None, None, :], align_corners=False)[:, 0, 0, 0]
        loss = (-activation + config.tv_weight * total_variation_per_sample(images)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0.0, 1.0)
        if step % config.snapshot_every == 0:
            snapshots.append(images.detach().clone())
        if step % 128 == 0:
            print(f"  step {step}/{config.steps} median activation {activation.median().item():.1f}", flush=True)
    return images.detach(), snapshots


class Trajectory(BaseModel):
    kind: str
    seed: int
    activation: list[float]
    nearest: list[int]
    nearest_tree: list[int]
    distance_over_tau: list[float]
    geodesic_to_cat: list[float]
    geodesic_to_snout: list[float]
    hops: list[int]
    """Edges between consecutive different nearest hits, in the order the trajectory meets them."""


def trajectory(ctx: Context, kind: str, seed: int, vectors: np.ndarray) -> Trajectory:
    nearest, distance = ctx.graph.nearest(vectors)
    changes = np.flatnonzero(np.diff(nearest) != 0)
    hops = ctx.graph.hops_between(nearest[changes], nearest[changes + 1])
    return Trajectory(
        kind=kind, seed=seed, activation=(vectors.sum(axis=1) + float(ctx.reader.bias)).tolist(),
        nearest=nearest.tolist(), nearest_tree=ctx.graph.trees.tree[nearest].tolist(),
        distance_over_tau=(distance / ctx.graph.tau).tolist(),
        geodesic_to_cat=ctx.graph.geodesic_to(nearest, ctx.graph.core_points(ctx.cat_trees)).tolist(),
        geodesic_to_snout=ctx.graph.geodesic_to(nearest, ctx.graph.core_points(ctx.snout_trees)).tolist(),
        hops=hops.tolist(),
    )


def lerp_vectors(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    alpha = np.linspace(0, 1, n)[:, None]
    return (1 - alpha) * start + alpha * end


def summarise(run: Trajectory, threshold: float) -> dict:
    activation = np.array(run.activation)
    in_regime = activation >= threshold
    ratio = np.array(run.distance_over_tau)
    hops = np.array(run.hops)
    real_hops = hops[hops > 0]
    return {
        "kind": run.kind, "seed": run.seed,
        "start_tree": run.nearest_tree[0], "final_tree": run.nearest_tree[-1],
        "start_activation": float(activation[0]), "final_activation": float(activation[-1]),
        "snapshots_in_hit_regime": int(in_regime.sum()),
        "median_distance_over_tau_in_regime": float(np.median(ratio[in_regime])) if in_regime.any() else None,
        "median_distance_over_tau": float(np.median(ratio)),
        "nearest_changes": len(hops),
        "distinct_nearest": len(set(run.nearest)),
        "median_hops_per_change": float(np.median(real_hops)) if len(real_hops) else None,
        "share_of_changes_to_a_graph_neighbour": float((hops == 1).mean()) if len(hops) else None,
        "final_geodesic_to_cat": run.geodesic_to_cat[-1], "final_geodesic_to_snout": run.geodesic_to_snout[-1],
        "trees_visited": list(dict.fromkeys(run.nearest_tree)),
    }


def plot_climb(runs: list[Trajectory], threshold: float, title: str, path: Path) -> None:
    colours = {"optimised": "#2563eb", "pixel_lerp": "#d97706", "hadamard_lerp": "#6b7280"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in runs:
        progress = np.linspace(0, 1, len(run.activation))
        style = {"color": colours[run.kind], "alpha": 0.9 if run.seed == 0 else 0.35, "lw": 1.5 if run.seed == 0 else 1}
        label = run.kind if run.seed == 0 else None
        axes[0].plot(progress, run.activation, label=label, **style)
        axes[1].plot(progress, run.distance_over_tau, label=label, **style)
        axes[2].plot(progress, run.geodesic_to_cat, label=label, **style)
    axes[0].axhline(threshold, color="black", lw=0.8, ls="--")
    axes[0].set_title("activation at the cell (dashed: hit threshold)")
    axes[1].set_title("distance to the nearest hit / tau")
    axes[1].set_yscale("log")
    axes[2].set_title("geodesic from the nearest hit to the cat cores")
    for axis in axes:
        axis.set_xlabel("progress along the path")
    axes[0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def climb_strip(ctx: Context, snapshots: list[torch.Tensor], run: Trajectory, row: int, col: int) -> Image.Image:
    frames = np.linspace(0, len(snapshots) - 1, STRIP_FRAMES).round().astype(int)
    top = [labelled(crop_around_cell(to_image(snapshots[f][0]), row, col), f"a {run.activation[f]:.0f}") for f in frames]
    bottom = [
        labelled(hit_crop(ctx, run.nearest[f]), f"t{run.nearest_tree[f]} d/tau {run.distance_over_tau[f]:.1f}")
        for f in frames
    ]
    sheet = Image.new("RGB", (CROP_PX * STRIP_FRAMES, CROP_PX * 2), "black")
    sheet.paste(strip(top), (0, 0))
    sheet.paste(strip(bottom), (0, CROP_PX))
    return sheet


class StartImage(BaseModel, arbitrary_types_allowed=True):
    name: str
    pixels: torch.Tensor
    row: int
    col: int


def run_climb(ctx: Context, start: StartImage, config: ClimbConfig, out_dir: Path) -> list[dict]:
    print(f"climb from {start.name} at cell ({start.row}, {start.col})", flush=True)
    final, snapshots = climb(ctx, start.pixels, start.row, start.col, config)
    stacked = torch.stack(snapshots)  # (n_snapshots, seeds, 3, h, w)
    runs: list[Trajectory] = []
    for seed in range(config.seeds):
        optimised = vectors_at(ctx.reader, stacked[:, seed], start.row, start.col)
        runs.append(trajectory(ctx, "optimised", seed, optimised))
        alpha = torch.linspace(0, 1, len(snapshots), device=ctx.device)[:, None, None, None]
        blended = (1 - alpha) * start.pixels + alpha * final[seed : seed + 1]
        runs.append(trajectory(ctx, "pixel_lerp", seed, vectors_at(ctx.reader, blended, start.row, start.col)))
        runs.append(trajectory(ctx, "hadamard_lerp", seed, lerp_vectors(optimised[0], optimised[-1], len(snapshots))))
        to_image(final[seed]).save(out_dir / f"{start.name}_final_seed{seed}.png")
    to_image(start.pixels[0]).save(out_dir / f"{start.name}_start.png")
    climb_strip(ctx, [s for s in stacked], runs[0], start.row, start.col).save(out_dir / f"{start.name}_strip_seed0.png")
    plot_climb(runs, ctx.graph.space.threshold, f"climb from {start.name}", out_dir / f"{start.name}_curves.png")
    (out_dir / f"{start.name}_trajectories.json").write_text(json.dumps([r.model_dump() for r in runs]))
    return [summarise(r, ctx.graph.space.threshold) | {"start": start.name} for r in runs]


def background_starts(ctx: Context, tree: int, count: int) -> list[StartImage]:
    """Hits of the background tree with a typical density, outside every core, each from a different image."""
    graph = ctx.graph
    in_any_core = np.zeros(len(graph.forest.counts), dtype=bool)
    in_any_core[np.concatenate(graph.cores)] = True
    members = np.flatnonzero((graph.trees.tree == tree) & ~in_any_core)
    typical = members[np.argsort(np.abs(graph.forest.kth_distance[members] - np.median(graph.forest.kth_distance[members])))]
    starts, used = [], set()
    for point in typical:
        sample_id, row, col = graph.hit_position(int(point))
        if sample_id in used or not (2 <= row <= 11 and 2 <= col <= 11):
            continue
        used.add(sample_id)
        pixels = to_pixels(display_image(ctx.image_dir / sample_id), ctx.device)
        starts.append(StartImage(name=f"background_t{tree}_{len(starts)}", pixels=pixels, row=row, col=col))
        if len(starts) == count:
            break
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-space", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--cat-image", type=Path, required=True)
    parser.add_argument("--sparse-image", action="append", default=[], help="name=path to a (3, h, w) .npy in [0, 1]")
    parser.add_argument("--cat-trees", required=True, help="comma-separated tree ids")
    parser.add_argument("--snout-trees", required=True, help="comma-separated tree ids")
    parser.add_argument("--background-tree", type=int, default=0)
    parser.add_argument("--background-starts", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    for field, info in ClimbConfig.model_fields.items():
        parser.add_argument(f"--climb-{field.replace('_', '-')}", type=info.annotation, default=info.default)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_lucent_inceptionv1(device)
    space = HitSpace.load(args.hit_space)
    ctx = Context(
        reader=HadamardReader(model, HADAMARD_CONV, CHANNEL), graph=HitGraph(space), image_dir=args.image_dir,
        cat_trees=[int(t) for t in args.cat_trees.split(",")], snout_trees=[int(t) for t in args.snout_trees.split(",")],
        device=device,
    )
    print(f"{len(space.activation)} hits, {len(ctx.graph.distinct_rows)} distinct, {len(space.trees.roots)} trees, tau {ctx.graph.tau:.4f}")

    place_dir = args.out / "place"
    place_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "cat_224": to_pixels(display_image(args.cat_image), device),
        "cat_448": to_pixels(short_side_image(args.cat_image, 448), device),
    }
    for entry in args.sparse_image:
        name, path = entry.split("=", 1)
        queries[name] = torch.from_numpy(np.load(path)).float()[None].to(device)
    placed = [place_query(ctx, name, pixels, place_dir) for name, pixels in queries.items()]
    (place_dir / "placed.json").write_text(json.dumps([p.model_dump() for p in placed], indent=1))
    for p in placed:
        print(f"place {p.name}: nearest tree {p.nearest_tree}, d/tau {p.distance_over_tau:.2f} "
              f"(percentile {p.nearest_distance_percentile:.0f}), ascent {len(p.ascent)} steps, "
              f"snout path {len(p.snout_path)} points through trees {list(dict.fromkeys(p.snout_path_trees))}", flush=True)

    climb_dir = args.out / "climb"
    climb_dir.mkdir(parents=True, exist_ok=True)
    climb_config = ClimbConfig(**{field: getattr(args, f"climb_{field}") for field in ClimbConfig.model_fields})
    grey = StartImage(name="grey", pixels=torch.full((1, 3, 224, 224), BACKGROUND_GREY, device=device), row=7, col=7)
    starts = [grey] + background_starts(ctx, args.background_tree, args.background_starts)
    summaries = [row for start in starts for row in run_climb(ctx, start, climb_config, climb_dir)]
    (climb_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
    (args.out / "run.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "tau": ctx.graph.tau}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
