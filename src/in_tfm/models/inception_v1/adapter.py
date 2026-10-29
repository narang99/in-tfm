from pathlib import Path

from ...attribution import compute_deeplift_relevance
from ...layers import submodule_getter
from ...neuron_report import AttrFn
from ...presenters import ImagePresenter
from ..base import reject_nnsight_wrapped
from ..stl_inception.adapter import tokens_in_layer
from .model import IMAGE_SIZE, AttributableInceptionV1, load_inception_v1
from .source import InceptionNormalization, InceptionSource


class InceptionV1Adapter:
    """The layer is fixed at construction, since the source must know how many positions it has."""

    attr_fn: AttrFn = staticmethod(compute_deeplift_relevance)

    def __init__(self, model: AttributableInceptionV1, layer_name: str, image_dir: Path) -> None:
        self.model = model
        self.layer_name = layer_name
        self.image_dir = image_dir

    @classmethod
    def from_pretrained(cls, layer_name: str, image_dir: Path) -> "InceptionV1Adapter":
        return cls(load_inception_v1(), layer_name, image_dir)

    def get_model(self) -> AttributableInceptionV1:
        return self.model

    def patch_for_attn_lrp(self) -> None:
        """Nothing to patch: there is no attention, and DeepLift hooks the modules itself."""
        reject_nnsight_wrapped(self.model)

    def make_source(self, samples: list[str]) -> InceptionSource:
        n_tokens = tokens_in_layer(self.model, submodule_getter(self.layer_name)(self.model), IMAGE_SIZE)
        return InceptionSource(self.image_dir, samples, n_tokens)

    def make_presenter(self, source: InceptionSource, clustered_label: str) -> ImagePresenter:
        return ImagePresenter(InceptionNormalization())
