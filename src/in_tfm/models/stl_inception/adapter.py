import torch
from datasets import load_dataset

from ...attribution import compute_deeplift_relevance
from ...layers import conv_getter
from ...presenters import ImagePresenter
from ..base import reject_nnsight_wrapped
from .model import StlInception
from .source import HubImages, ImageDataset, StlNormalization, StlSource


def load_model(checkpoint: str | None) -> StlInception:
    model = StlInception()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    return model.eval()

IMAGE_SIZE = 96


def tokens_in_layer(model: torch.nn.Module, layer: torch.nn.Module) -> int:
    """Output positions per image, found by running one blank image through the model.

    The image goes to wherever the model is, since the pipeline moves the shared module before it asks for a source.
    """
    counts: list[int] = []
    handle = layer.register_forward_hook(lambda module, inputs, output: counts.append(output.shape[-2] * output.shape[-1]))
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=next(model.parameters()).device))
    finally:
        handle.remove()
    return counts[0]


class StlInceptionAdapter:
    """The layer is fixed at construction, since the source must know how many positions it has."""

    attr_fn = staticmethod(compute_deeplift_relevance)

    def __init__(self, model: StlInception, layer_name: str, dataset: ImageDataset, split: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self.dataset = dataset
        self.split = split

    @classmethod
    def from_pretrained(
        cls, layer_name: str, dataset: str, split: str, checkpoint: str | None = None
    ) -> "StlInceptionAdapter":
        model = load_model(checkpoint)
        return cls(model, layer_name, HubImages(load_dataset(dataset, split=split)), split)

    def get_model(self) -> StlInception:
        return self.model

    def patch_for_attn_lrp(self) -> None:
        """Nothing to patch: there is no attention, and DeepLift hooks the modules itself."""
        reject_nnsight_wrapped(self.model)

    def make_source(self, samples: list[int]) -> StlSource:
        n_tokens = tokens_in_layer(self.model, conv_getter(self.layer_name)(self.model))
        return StlSource(self.dataset, samples, self.split, n_tokens)

    def make_presenter(self, source: StlSource, clustered_label: str) -> ImagePresenter:
        return ImagePresenter(StlNormalization())
