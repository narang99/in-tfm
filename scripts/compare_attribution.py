#!/usr/bin/env python
"""Compare our DeepLift against captum's Integrated Gradients on the STL mini-Inception.

    uv run python scripts/compare_attribution.py --n-images 32 --ig-steps '[8,32,128]'

Prints one row per IG step count, with the time of both methods and how well they agree.
"""

import torch
from datasets import load_dataset

from in_tfm.attribution_comparison import ComparisonConfig, StepResult, measure_batch, merge
from in_tfm.layers import conv_getter
from in_tfm.models.stl_inception.adapter import load_model
from in_tfm.models.stl_inception.source import HubImages, eval_transform


def load_images(config: ComparisonConfig) -> list[torch.Tensor]:
    dataset = HubImages(load_dataset(config.dataset, split=config.split))
    return [eval_transform(dataset[i][0]) for i in range(config.n_images)]


def batches(images: list[torch.Tensor], size: int, device: str) -> list[torch.Tensor]:
    return [torch.stack(images[i : i + size]).to(device) for i in range(0, len(images), size)]


def print_table(results: list[StepResult]) -> None:
    print("steps  ig_s  dl_s  speedup  pearson  top_k  ig_gap  dl_gap")
    for r in results:
        print(
            f"{r.ig_steps:5d} {r.ig_seconds:5.2f} {r.deeplift_seconds:5.2f} {r.ig_seconds / r.deeplift_seconds:7.1f}x "
            f"{r.pearson:8.3f} {r.top_k_overlap:6.3f} {r.ig_completeness_gap:7.3f} {r.deeplift_completeness_gap:7.4f}"
        )


def main() -> None:
    config = ComparisonConfig(_cli_parse_args=True)
    print(config)
    model = load_model(str(config.checkpoint) if config.checkpoint else None).to(config.device)
    layer = conv_getter(config.layer_name)(model)
    chunks = batches(load_images(config), config.batch_size, config.device)
    measure_batch(model, layer, chunks[0][:2], config.model_copy(update={"ig_steps": [2]}))
    per_batch = [measure_batch(model, layer, chunk, config) for chunk in chunks]
    results = merge(per_batch, [len(chunk) for chunk in chunks])
    print_table(results)
    if config.out_path is not None:
        config.out_path.write_text("[" + ",".join(r.model_dump_json() for r in results) + "]")


if __name__ == "__main__":
    main()
