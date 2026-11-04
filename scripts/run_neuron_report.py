#!/usr/bin/env python
"""Neuron reports for any onboarded model: a language model over a text corpus, MNIST, or InceptionV1 over an image folder.

The model name decides the modality, and everything after loading the samples is
`in_tfm.neuron_run.run_neuron_reports`. Settings are in `in_tfm.run_config.RunConfig`, with
defaults from `--config <yaml>` (see `configs/`) and flags on top:

    python scripts/run_neuron_report.py --config configs/gemma3_l11_down_proj_peaks.yaml --neurons 90
    python scripts/run_neuron_report.py --config configs/inception_v1_mixed4e_peaks.yaml

Clustering picks a GPU backend when one is available, see in_tfm.clustering. On CPU, keep the
corpus small: hit count grows linearly with it while clustering cost grows with the square of
the hit count.
"""

import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from in_tfm.models import ModelAdapter
from in_tfm.models.attn_only_2l import AttnOnly2LAdapter, TinyAttnOnlyAdapter
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.models.inception_v1.adapter import InceptionV1Adapter
from in_tfm.models.inception_v1.source import image_paths, spread_indices
from in_tfm.models.mnist_cnn.adapter import MnistCnnAdapter, MnistCnnSmallAdapter
from in_tfm.models.mnist_cnn.source import first_per_class
from in_tfm.models.qwen3 import Qwen3Adapter
from in_tfm.neuron_run import run_neuron_reports, timed
from in_tfm.run_config import ModelName, RunConfig, load_run_config
from in_tfm.sources.wikitext import load_texts

TINY_ATTN_ONLY_1L_CHECKPOINT = Path("checkpoints/tiny_attn_only_1l")


def load_adapter(config: RunConfig) -> ModelAdapter[Any, Any]:
    match config.model:
        case ModelName.GEMMA3_270M:
            return Gemma3Adapter.from_pretrained(
                config.model.value, config.text.max_length
            )
        case ModelName.MARIN_32B:
            return Qwen3Adapter.from_pretrained(
                config.model.value,
                config.layer_name,
                config.text.max_length,
                device=config.device,
            )
        case ModelName.ATTN_ONLY_2L:
            return AttnOnly2LAdapter.from_pretrained(config.text.max_length)
        case ModelName.TINY_ATTN_ONLY_1L:
            return TinyAttnOnlyAdapter.from_checkpoint(
                TINY_ATTN_ONLY_1L_CHECKPOINT, config.text.max_length
            )
        case ModelName.MNIST_CNN:
            return MnistCnnAdapter.from_pretrained(
                config.layer_name,
                config.mnist.dataset,
                config.mnist.split,
                config.mnist.checkpoint,
            )
        case ModelName.MNIST_CNN_SMALL:
            return MnistCnnSmallAdapter.from_pretrained(
                config.layer_name,
                config.mnist.dataset,
                config.mnist.split,
                config.mnist.checkpoint,
            )
        case ModelName.INCEPTION_V1:
            return InceptionV1Adapter.from_pretrained(
                config.layer_name, config.inception.image_dir
            )


def load_inception_samples(config: RunConfig) -> list[str]:
    paths = image_paths(config.inception.image_dir)
    if config.inception.n_images is None:
        return paths
    return [
        paths[i]
        for i in spread_indices(len(paths), min(config.inception.n_images, len(paths)))
    ]


def load_samples(config: RunConfig) -> list[str] | list[int]:
    if config.modality == "inception":
        return load_inception_samples(config)
    if config.modality == "mnist":
        labels = load_dataset(config.mnist.dataset, split=config.mnist.split)["label"]
        return first_per_class(labels, config.mnist.images_per_class)
    texts = load_texts(config.text, config.seed)
    print(f"loaded {len(texts)} {config.text.unit}s")
    return texts


def main() -> None:
    pipeline_start = time.perf_counter()
    config = load_run_config(sys.argv[1:])

    with timed("load samples"):
        samples = load_samples(config)
    with timed("load model"):
        adapter = load_adapter(config)

    run_neuron_reports(config, adapter, samples)
    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
