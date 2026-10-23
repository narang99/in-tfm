from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ...sources import ModelBatch, SampleId
from .model import IMAGE_SIZE, INPUT_MEAN, INPUT_STD

IMAGE_KEY = "x"
"""Named after `AttributableInceptionV1.forward`'s argument, which the batch is splatted into."""

RESIZE_SIZE = 256

eval_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([INPUT_MEAN] * 3, [INPUT_STD] * 3),
    ]
)


def image_paths(image_dir: Path) -> list[str]:
    """Every jpg under `image_dir`, relative to it and sorted.

    - Sorted explicitly, since the filesystem's own order differs between machines.
    - With one folder per class this is class-sorted, so take a subset with `spread_indices`, not a prefix.
    """
    return sorted(path.relative_to(image_dir).as_posix() for path in image_dir.rglob("*.jpg"))


class InceptionNormalization:
    """Lets `ImagePresenter` undo the normalisation, the same way it does for an image processor."""

    image_mean = [INPUT_MEAN] * 3
    image_std = [INPUT_STD] * 3


class InceptionSource:
    """Images of one directory, picked by their path relative to it, which is also the sample id."""

    def __init__(self, image_dir: Path, relative_paths: Sequence[str], n_tokens: int) -> None:
        self.image_dir = image_dir
        self.relative_paths = list(relative_paths)
        self.n_tokens = n_tokens
        """Positions the traced layer has per image. The mask is all True, but its width must match
        the activations, since hit selection indexes both with the same positions."""

    def sample_ids(self) -> Sequence[SampleId]:
        return list(self.relative_paths)

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        images = torch.stack([eval_transform(Image.open(self.image_dir / i).convert("RGB")) for i in ids])
        return ModelBatch(
            kwargs={IMAGE_KEY: images},
            grad_leaf_key=IMAGE_KEY,
            valid_mask=torch.ones(len(ids), self.n_tokens, dtype=torch.bool),
        )
