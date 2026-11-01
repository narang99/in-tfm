#!/usr/bin/env python
"""Sparse versions of images that keep one InceptionV1 channel's strong activations.

    uv run --extra inception python scripts/run_sparse_feature_vis.py cat.jpg --neuron mixed4e:55 --out outputs/sparse_vis

Output:

- `<out>/<image>.png` is the seed-averaged image.
- `<out>/intermediates/<image>/` holds the input, the map, the target, every seed, and `summary.json`.
  - `*_maps.png` is `target | produced`, both scaled by the target's peak.
  - `std_across_seeds.png` is bright where the seeds disagree.
"""

import argparse
import json
from pathlib import Path
from typing import Literal, get_args, get_origin

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from in_tfm.device import default_device
from in_tfm.sparse_feature_vis import (
    ChannelReader,
    HadamardObjective,
    HadamardReader,
    MapObjective,
    PeakHadamard,
    SparseTarget,
    SparseVisConfig,
    build_target,
    channel_maps,
    load_lucent_inceptionv1,
    optimise_all_seeds,
    peak_hadamard,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAP_CELL_PX = 16


def load_rgb(path: Path, short_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = short_side / min(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def to_grayscale(values: np.ndarray) -> Image.Image:
    peak = values.max()
    scaled = values / peak if peak > 0 else values
    return Image.fromarray((scaled * 255).round().astype(np.uint8), mode="L")


def save_rgb(pixels: torch.Tensor, path: Path) -> None:
    array = pixels.permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array.clip(0, 1) * 255).round().astype(np.uint8)).save(path)


def save_map_pair(produced: np.ndarray, target: np.ndarray, path: Path) -> None:
    peak = target.max()
    divider = np.full((target.shape[0], 1), peak)
    pair = np.concatenate([target, divider, np.clip(produced, 0, peak)], axis=1)
    size = (pair.shape[1] * MAP_CELL_PX, pair.shape[0] * MAP_CELL_PX)
    to_grayscale(pair).resize(size, resample=Image.NEAREST).save(path)


def per_seed_rows(images: torch.Tensor, maps: torch.Tensor, target: torch.Tensor) -> list[dict]:
    mse = ((maps - target) ** 2).mean(dim=(1, 2, 3))
    return [
        {
            "seed": seed,
            "map_mse": mse[seed].item(),
            "map_peak": maps[seed].max().item(),
            "exact_zero_frac": (images[seed].amax(0) == 0).float().mean().item(),
            "mean_abs": images[seed].abs().mean().item(),
        }
        for seed in range(len(images))
    ]


def save_target(sparse_target: SparseTarget, image: Image.Image, work_dir: Path) -> None:
    np.save(work_dir / "activation_map.npy", sparse_target.activation_map[0, 0].cpu().numpy())
    target = sparse_target.target[0, 0].cpu().numpy()
    np.save(work_dir / "target.npy", target)
    to_grayscale(target).resize(image.size, Image.NEAREST).save(work_dir / "target.png")


def save_seeds(images: torch.Tensor, maps: torch.Tensor, target: np.ndarray, seeds_dir: Path) -> None:
    for seed in range(len(images)):
        save_rgb(images[seed], seeds_dir / f"seed_{seed:02d}.png")
        save_map_pair(maps[seed, 0].cpu().numpy(), target, seeds_dir / f"seed_{seed:02d}_maps.png")


def process_image(read_channel: ChannelReader, image_path: Path, config: SparseVisConfig, out_dir: Path, device: torch.device) -> dict:
    name = image_path.stem
    work_dir = out_dir / "intermediates" / name
    (work_dir / "seeds").mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path, config.size)
    image.save(work_dir / "input.png")
    reference = image_to_tensor(image).to(device)
    sparse_target = build_target(read_channel, reference, config)
    save_target(sparse_target, image, work_dir)

    summary: dict = {
        "image": name,
        "map_peak": sparse_target.activation_map.max().item(),
        "threshold": sparse_target.threshold,
        "target_cells": sparse_target.cell_count,
    }
    if sparse_target.cell_count == 0:
        summary["skipped"] = f"map peak {summary['map_peak']:.1f} is below threshold {sparse_target.threshold:g}"
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary

    target = sparse_target.target
    if config.objective == "hadamard":
        read_hadamard = HadamardReader(read_channel.model, config.hadamard_conv, read_channel.channel)
        peak = peak_hadamard(read_hadamard, reference)
        summary["hadamard_peak"] = {"row": peak.row, "col": peak.col, "activation": peak.activation}
        objective = HadamardObjective(read_hadamard, peak, config.hadamard_weight)
    else:
        objective = MapObjective(read_channel, target)
    images = optimise_all_seeds(objective, reference, config)
    maps = channel_maps(read_channel, images, config.seed_batch)
    save_seeds(images, maps, target[0, 0].cpu().numpy(), work_dir / "seeds")

    mean_image = images.mean(0)
    mean_map = channel_maps(read_channel, mean_image[None], 1)
    save_rgb(mean_image, out_dir / f"{name}.png")
    np.save(work_dir / "mean_image.npy", mean_image.cpu().numpy())
    to_grayscale(images.std(0).amax(0).cpu().numpy()).save(work_dir / "std_across_seeds.png")
    save_map_pair(mean_map[0, 0].cpu().numpy(), target[0, 0].cpu().numpy(), work_dir / "mean_image_maps.png")

    summary["seeds"] = per_seed_rows(images, maps, target)
    summary["mean_image"] = {
        "map_mse": ((mean_map - target) ** 2).mean().item(),
        "map_peak": mean_map.max().item(),
        "exact_zero_frac": (mean_image.amax(0) == 0).float().mean().item(),
        "mean_abs": mean_image.abs().mean().item(),
    }
    if config.objective == "hadamard":
        add_hadamard_rows(summary, read_hadamard, peak, images, mean_image, config.seed_batch)
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def hadamard_fit(read_hadamard: HadamardReader, peak: PeakHadamard, images: torch.Tensor, batch_size: int) -> list[dict]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            vectors = read_hadamard(images[start : start + batch_size])[:, :, peak.row, peak.col]
            for vector in vectors:
                rows.append({
                    "hadamard_mse": ((vector - peak.target) ** 2).mean().item(),
                    "hadamard_cosine": F.cosine_similarity(vector, peak.target, dim=0).item(),
                    "peak_activation": (vector.sum() + read_hadamard.bias).item(),
                })
    return rows


def add_hadamard_rows(
    summary: dict, read_hadamard: HadamardReader, peak: PeakHadamard, images: torch.Tensor, mean_image: torch.Tensor, batch_size: int
) -> None:
    for seed_row, fit_row in zip(summary["seeds"], hadamard_fit(read_hadamard, peak, images, batch_size)):
        seed_row.update(fit_row)
    summary["mean_image"].update(hadamard_fit(read_hadamard, peak, mean_image[None], 1)[0])


def find_images(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            paths.append(path)
    return paths


def print_result(summary: dict) -> None:
    if "skipped" in summary:
        print(f"{summary['image']}: skipped ({summary['skipped']})", flush=True)
        return
    seed_mse = np.median([row["map_mse"] for row in summary["seeds"]])
    mean = summary["mean_image"]
    print(
        f"{summary['image']}: {summary['target_cells']} target cells, seed map MSE median {seed_mse:.1f}, "
        f"mean-image map MSE {mean['map_mse']:.1f}, mean-image zero frac {mean['exact_zero_frac']:.2f}",
        flush=True,
    )


def parse_args() -> tuple[argparse.Namespace, SparseVisConfig]:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+", help="image files or folders of images")
    parser.add_argument("--neuron", default="mixed4e:55", help="lucent layer:channel")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    for field, info in SparseVisConfig.model_fields.items():
        flag = f"--{field.replace('_', '-')}"
        if get_origin(info.annotation) is Literal:
            parser.add_argument(flag, choices=get_args(info.annotation), default=info.default)
        else:
            parser.add_argument(flag, type=info.annotation, default=info.default)
    args = parser.parse_args()
    config = SparseVisConfig(**{field: getattr(args, field) for field in SparseVisConfig.model_fields})
    return args, config


def main() -> None:
    args, config = parse_args()
    device = torch.device(args.device)
    layer, channel = args.neuron.split(":")
    read_channel = ChannelReader(load_lucent_inceptionv1(device), layer, int(channel))
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [process_image(read_channel, path, config, args.out, device) for path in find_images(args.inputs)]
    for summary in summaries:
        print_result(summary)
    run = {"neuron": args.neuron, "config": config.model_dump(), "images": summaries}
    (args.out / "intermediates" / "run_summary.json").write_text(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
