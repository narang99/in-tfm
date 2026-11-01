#!/usr/bin/env python
"""Every hit of one InceptionV1 neuron with its density-peak forest, saved as one `hit_space.npz`.

- It runs the report pipeline's scan, hit selection, gather and forest, and stops before any page.
- It also draws each tree's peak core as crops around the hit positions, to tell what each tree responds to.

    uv run --extra inception python scripts/build_hit_space.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --hit-space-out reports/hit_space
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from in_tfm.density_peaks.probabilities import peak_cores
from in_tfm.hit_selection.selection import select_hits
from in_tfm.hit_space import HitSpace
from in_tfm.models.inception_v1.adapter import InceptionV1Adapter
from in_tfm.models.inception_v1.source import image_paths
from in_tfm.models.stl_inception.source import spread_indices
from in_tfm.neuron_run import clustered_rows, density_peak_trees, forest_device, layer_getter_for, prepare_run, timed
from in_tfm.run_config import DensityPeaksConfig, RunConfig, load_run_config

CROP_PX = 96
CELL_PX = 16
CORE_CROPS_PER_TREE = 8
TREES_PER_SHEET = 14


def inception_samples(config: RunConfig) -> list[str]:
    paths = image_paths(config.inception.image_dir)
    if config.inception.n_images is None:
        return paths
    return [paths[i] for i in spread_indices(len(paths), min(config.inception.n_images, len(paths)))]


def build(config: RunConfig) -> HitSpace:
    if len(config.neuron_idxs) != 1 or config.polarity != "positive":
        raise ValueError("one neuron, positive polarity")
    strategy = config.clustering.strategy
    if not isinstance(strategy, DensityPeaksConfig):
        raise ValueError("the config must use density peaks")
    neuron = config.neuron_idxs[0]
    run = prepare_run(config, InceptionV1Adapter.from_pretrained(config.layer_name, config.inception.image_dir), inception_samples(config))
    with timed("scan"):
        scan = run.capture.scan()
    selection = select_hits(scan, neuron, "positive", config.hits, config.seed)
    with timed("gather"):
        gathered = run.capture.gather(selection.batch_idx, selection.token_idx)
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)
    hadamard = clustered_rows(gathered.inputs, weight, neuron, config.clustering)
    with timed("forest"):
        forest, trees = density_peak_trees(hadamard, config.clustering, strategy, forest_device(config.device))
    grid_width = int(round(np.sqrt(scan.activations.shape[1])))
    return HitSpace(
        hadamard=hadamard.astype(np.float32), activation=selection.activations.astype(np.float32),
        sample_ids=list(scan.sample_ids), batch_idx=selection.batch_idx, token_idx=selection.token_idx,
        grid_width=grid_width, threshold=float(selection.threshold), distance=config.clustering.distance,
        normalise=config.clustering.normalise, peak_share=strategy.peak_share, forest=forest, trees=trees,
    )


def display_image(image_dir: Path, sample_id: str) -> Image.Image:
    """What the network saw: resized to 256 and centre-cropped to 224, as `source.eval_transform` does."""
    image = Image.open(image_dir / sample_id).convert("RGB")
    scale = 256 / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    return image.crop((left, top, left + 224, top + 224))


def crop_around_cell(image: Image.Image, row: int, col: int, cell_px: int = CELL_PX, crop_px: int = CROP_PX) -> Image.Image:
    centre_x, centre_y = col * cell_px + cell_px // 2, row * cell_px + cell_px // 2
    half = crop_px // 2
    return image.crop((centre_x - half, centre_y - half, centre_x + half, centre_y + half))


def core_sheet(space: HitSpace, image_dir: Path, tree_ids: list[int], cores: list[np.ndarray]) -> Image.Image:
    sheet = Image.new("RGB", (CROP_PX * (CORE_CROPS_PER_TREE + 1), CROP_PX * len(tree_ids)), "black")
    draw = ImageDraw.Draw(sheet)
    first_hit = {point: hit for hit, point in reversed(list(enumerate(space.forest.distinct_of_hit)))}
    for slot, tree in enumerate(tree_ids):
        hits_in_tree = int(space.forest.counts[space.trees.tree == tree].sum())
        draw.text((4, slot * CROP_PX + 4), f"tree {tree}\n{hits_in_tree} hits\ncore {len(cores[tree])}", fill="yellow")
        densest = cores[tree][np.argsort(space.forest.kth_distance[cores[tree]])][:CORE_CROPS_PER_TREE]
        for column, point in enumerate(densest, start=1):
            hit = first_hit[point]
            row, col = divmod(int(space.token_idx[hit]), space.grid_width)
            image = display_image(image_dir, space.sample_ids[space.batch_idx[hit]])
            sheet.paste(crop_around_cell(image, row, col), (column * CROP_PX, slot * CROP_PX))
    return sheet


def write_core_sheets(space: HitSpace, image_dir: Path, out_dir: Path) -> None:
    cores = peak_cores(space.forest, space.trees, space.peak_share)
    n_trees = len(space.trees.roots)
    for start in range(0, n_trees, TREES_PER_SHEET):
        tree_ids = list(range(start, min(start + TREES_PER_SHEET, n_trees)))
        core_sheet(space, image_dir, tree_ids, cores).save(out_dir / f"cores_{start:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-space-out", type=Path, required=True)
    args, rest = parser.parse_known_args()
    config = load_run_config(rest)
    sys.argv = sys.argv[:1]
    space = build(config)
    args.hit_space_out.mkdir(parents=True, exist_ok=True)
    space.save(args.hit_space_out / "hit_space.npz")
    write_core_sheets(space, config.inception.image_dir, args.hit_space_out)
    summary = {
        "n_hits": len(space.activation), "n_distinct": len(space.forest.counts), "n_trees": len(space.trees.roots),
        "threshold": space.threshold,
        "hits_per_tree": [int(space.forest.counts[space.trees.tree == t].sum()) for t in range(len(space.trees.roots))],
    }
    (args.hit_space_out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
