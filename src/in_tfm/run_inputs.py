"""The adapter and samples a run config names, shared by the neuron and dependency report scripts.

- A dependency report must see the same samples as the neuron report it reads, so both load them here.
"""

from typing import Any

from datasets import load_dataset

from .models import ModelAdapter
from .models.attn_only_2l import (
    AttnOnly2LAdapter,
    TinyAttnOnly1LAdapter,
    TinyAttnOnly1LD128Adapter,
)
from .models.gemma3 import Gemma3Adapter
from .models.inception_v1.adapter import InceptionV1Adapter
from .models.inception_v1.source import image_paths, spread_indices
from .models.mnist_cnn.adapter import MnistCnnAdapter, MnistCnnSmallAdapter
from .models.mnist_cnn.source import first_per_class
from .run_config import ModelName, RunConfig
from .sources.wikitext import load_texts


def load_adapter(config: RunConfig) -> ModelAdapter[Any, Any]:
    match config.model:
        case ModelName.GEMMA3_270M:
            return Gemma3Adapter.from_pretrained(
                config.model.value, config.text.max_length
            )
        case ModelName.ATTN_ONLY_2L:
            return AttnOnly2LAdapter.from_pretrained(config.text.max_length)
        case ModelName.TINY_ATTN_ONLY_1L:
            return TinyAttnOnly1LAdapter.from_pretrained(config.text.max_length)
        case ModelName.TINY_ATTN_ONLY_1L_D128:
            return TinyAttnOnly1LD128Adapter.from_pretrained(config.text.max_length)
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
