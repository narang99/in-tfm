from collections.abc import Sequence

import torch
from torchvision.transforms.functional import to_tensor

from ...sources import ModelBatch, SampleId

IMAGE_KEY = "x"
"""Named after `MnistCnn.forward`'s argument, which the batch is splatted into."""


class Unnormalized:
    """The model reads raw [0, 1] pixels, so `ImagePresenter` has nothing to undo."""

    image_mean = [0.0]
    image_std = [1.0]


def first_per_class(labels: Sequence[int], per_class: int) -> list[int]:
    """The first `per_class` indices of every label, in dataset order, so every digit is equally represented."""
    taken: dict[int, int] = {}
    indices = []
    for index, label in enumerate(labels):
        if taken.get(label, 0) < per_class:
            taken[label] = taken.get(label, 0) + 1
            indices.append(index)
    return indices


class MnistSource:
    """Images of one MNIST split, picked by dataset index. The sample id is `{split}:{index}`."""

    def __init__(self, split_rows, indices: Sequence[int], split: str, n_tokens: int) -> None:
        self.split_rows = split_rows
        """A hub split with a grayscale `image` column, or any list of such rows."""
        self.indices = list(indices)
        self.split = split
        self.n_tokens = n_tokens
        """Positions the traced layer has per image. The mask is all True, but its width must match
        the activations, since hit selection indexes both with the same positions."""

    def sample_ids(self) -> Sequence[SampleId]:
        return [f"{self.split}:{i}" for i in self.indices]

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        images = torch.stack([to_tensor(self.split_rows[int(i.split(":")[1])]["image"]) for i in ids])
        return ModelBatch(
            kwargs={IMAGE_KEY: images},
            grad_leaf_key=IMAGE_KEY,
            valid_mask=torch.ones(len(ids), self.n_tokens, dtype=torch.bool),
        )
