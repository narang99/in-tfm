"""Two trained MNIST networks, each its own model, sharing one adapter implementation.

- `MnistCnnAdapter` is the 8/16 channel network, the smallest one above 98% accuracy.
- `MnistCnnSmallAdapter` is the 2/4 channel network, the smallest one, at about 94% accuracy.
- Hits are explained with DeepLift against a blank image, see `MnistCnn`.
"""

from pathlib import Path
from typing import ClassVar

import torch
from datasets import load_dataset

from ...attribution import compute_deeplift_relevance
from ...layers import conv_getter
from ...mnist_training import load_checkpoint
from ...neuron_report import AttrFn
from ...presenters import ImagePresenter
from ..base import reject_nnsight_wrapped
from .model import IMAGE_SIZE, MnistCnn, MnistCnnChannels
from .source import MnistSource, Unnormalized


def positions_in_output(output: torch.Tensor) -> int:
    """A conv's grid cells, or 1 for the linear head, whose output has no position axis."""
    return output.shape[-2] * output.shape[-1] if output.ndim == 4 else 1


def positions_in_layer(model: torch.nn.Module, layer: torch.nn.Module) -> int:
    """Output positions per image, found by running one blank image on the model's own device."""
    counts: list[int] = []
    handle = layer.register_forward_hook(lambda module, inputs, output: counts.append(positions_in_output(output)))
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=next(model.parameters()).device))
    finally:
        handle.remove()
    return counts[0]


class MnistAdapter:
    """Subclasses name a trained network by its widths and where its checkpoint lives."""

    channels: ClassVar[MnistCnnChannels]
    default_checkpoint: ClassVar[Path]
    attr_fn: AttrFn = staticmethod(compute_deeplift_relevance)

    def __init__(self, model: MnistCnn, layer_name: str, split_rows, split: str) -> None:
        if model.channels != self.channels:
            raise ValueError(f"{type(self).__name__} is the {self.channels} network, got {model.channels}")
        self.model = model
        self.layer_name = layer_name
        self.split_rows = split_rows
        self.split = split

    @classmethod
    def from_pretrained(cls, layer_name: str, dataset: str, split: str, checkpoint: Path | None = None):
        model = load_checkpoint(checkpoint or cls.default_checkpoint)
        return cls(model, layer_name, load_dataset(dataset, split=split), split)

    def get_model(self) -> MnistCnn:
        return self.model

    def patch_for_attn_lrp(self) -> None:
        """Nothing to patch: there is no attention, and DeepLift hooks the modules itself."""
        reject_nnsight_wrapped(self.model)

    def make_source(self, samples: list[int]) -> MnistSource:
        n_tokens = positions_in_layer(self.model, conv_getter(self.layer_name)(self.model))
        return MnistSource(self.split_rows, samples, self.split, n_tokens)

    def make_presenter(self, source: MnistSource, clustered_label: str) -> ImagePresenter:
        return ImagePresenter(Unnormalized())


class MnistCnnAdapter(MnistAdapter):
    channels = MnistCnnChannels(conv1=8, conv2=16)
    default_checkpoint = Path("checkpoints/mnist_cnn/final.pt")


class MnistCnnSmallAdapter(MnistAdapter):
    channels = MnistCnnChannels(conv1=2, conv2=4)
    default_checkpoint = Path("checkpoints/mnist_cnn_2_4/final.pt")
