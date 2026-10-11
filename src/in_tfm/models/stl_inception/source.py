from collections.abc import Sequence
from typing import Protocol

import torch
from PIL import Image
from torchvision import transforms

from ...sources import ModelBatch, SampleId

STL_MEAN = (0.4467, 0.4398, 0.4066)
STL_STD = (0.2603, 0.2566, 0.2713)
"""The statistics of STL-10's training images. The model is trained on inputs normalised with them."""

IMAGE_KEY = "x"
"""Named after `StlInception.forward`'s argument, which the batch is splatted into."""

eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(STL_MEAN, STL_STD)])


class ImageDataset(Protocol):
    """What `HubImages` offers, so tests can hand in a few synthetic images."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Image.Image, int]: ...


class HubImages:
    """A Hugging Face image classification split, as `(image, label)` pairs."""

    def __init__(self, split) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        row = self.split[index]
        return row["image"].convert("RGB"), row["label"]


class StlNormalization:
    """Lets `ImagePresenter` undo the normalisation, the same way it does for an image processor."""

    image_mean = list(STL_MEAN)
    image_std = list(STL_STD)


class StlSource:
    """Images of one STL-10 split, picked by dataset index. The sample id is `{split}:{index}`."""

    def __init__(self, dataset: ImageDataset, indices: Sequence[int], split: str, n_tokens: int) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.split = split
        self.n_tokens = n_tokens
        """Positions the traced layer has per image. The mask is all True, but its width must match
        the activations, since hit selection indexes both with the same positions."""

    def sample_ids(self) -> Sequence[SampleId]:
        return [f"{self.split}:{i}" for i in self.indices]

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        images = torch.stack([eval_transform(self.dataset[int(i.split(":")[1])][0]) for i in ids])
        return ModelBatch(
            kwargs={IMAGE_KEY: images},
            grad_leaf_key=IMAGE_KEY,
            valid_mask=torch.ones(len(ids), self.n_tokens, dtype=torch.bool),
        )


def spread_indices(total: int, count: int) -> list[int]:
    """`count` evenly spaced indices into a split of `total` images.

    - Index `k` is `(k * total) // count`, so the gap between neighbours is `total / count`.
      - The multiply comes first so the arithmetic stays in integers, with no float rounding.
      - The indices are distinct as long as `count <= total`, since the gap is then at least 1.
    - The hub's STL-10 splits are sorted by class, so the first `count` images are all one class.
      - Evenly spaced picks take every class in proportion.
      - For the 8000 test images, `count=500` gives `0, 16, 32, ...`, which is 50 images of each of the 10 classes.
    """
    return [(index * total) // count for index in range(count)]
