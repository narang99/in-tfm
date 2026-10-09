from functools import partial
from pathlib import Path

import torch

from .. import compat  # noqa: F401  (must precede any lxt import, see compat.py)
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import dropout_forward, layer_norm_forward, non_linear_forward, patch_method
from rad_dino import RadDino
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
import transformers.models.dinov2.modeling_dinov2 as modeling_dinov2
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

from ..attribution import compute_attnlrp_relevance, patch_eager_attention
from ..presenters import ImagePresenter
from ..sources import DicomSource
from .base import reject_nnsight_wrapped

RAD_DINO_NAME = "microsoft/rad-dino"


def patch_dinov2_for_attn_lrp(model: Dinov2Model) -> None:
    """Monkey-patch a RadDino/Dinov2 model in place for AttnLRP.

    lxt.efficient has no built-in support for Dinov2Model, but Dinov2's attention already uses
    the same ALL_ATTENTION_FUNCTIONS/eager_attention_forward interface as Llama/Qwen/Gemma, so
    the primitives lxt.efficient.patches ships for those models cover Dinov2 too:
      - LayerNorm -> identity rule (stop-gradient through mean/std)
      - the MLP activation -> identity rule (gradient*input reproduces f(x) exactly)
      - Dropout -> identity, in case .train() is ever used
      - attention (Q@K^T, attn@V) -> uniform rule, via patch_dinov2_attention
    nn.Linear/nn.Conv2d and Dinov2's LayerScale (multiply by a learned, input-independent
    vector) need no patch: plain gradient*input is already a valid LRP rule for anything
    linear in the input.
    """
    # Discovered from the live model rather than hardcoded, so this keeps working if the
    # checkpoint's config.hidden_act ever resolves to a different ACT2FN class.
    activation_cls = type(model.encoder.layer[0].mlp.activation_fn)

    patch_map = {
        torch.nn.LayerNorm: partial(patch_method, layer_norm_forward),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        activation_cls: partial(patch_method, non_linear_forward, keep_original=True),
        modeling_dinov2: patch_eager_attention,
    }

    # NOTE: LayerNorm/Dropout are patched at the class level, i.e. process-wide - any other
    # model sharing this process will also get LRP-flavored LayerNorm/Dropout.
    model.config._attn_implementation = "eager"
    monkey_patch(model, patch_map=patch_map, verbose=True)


class RadDinoAdapter:
    """A single instance serves capture and attribution, so the weights are loaded once."""

    attr_fn = staticmethod(compute_attnlrp_relevance)

    def __init__(self, model: Dinov2Model, processor: BaseImageProcessor) -> None:
        self.model = model
        self.processor = processor
        self._patched = False

    @classmethod
    def from_pretrained(cls) -> "RadDinoAdapter":
        return cls(RadDino().model, AutoImageProcessor.from_pretrained(RAD_DINO_NAME))

    def get_model(self) -> Dinov2Model:
        return self.model

    def patch_for_attn_lrp(self) -> None:
        if self._patched:
            return
        reject_nnsight_wrapped(self.model)
        patch_dinov2_for_attn_lrp(self.model)
        self._patched = True

    def make_source(self, samples: list[Path]) -> DicomSource:
        return DicomSource(samples, self.processor)

    def make_presenter(self, source: DicomSource, clustered_label: str) -> ImagePresenter:
        return ImagePresenter(self.processor)
