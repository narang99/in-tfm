"""What the pipeline needs from a model, and nothing else.

- The adapter owns the model, its tokenizer or processor, and the choices that depend on both.
- Layer getters are deliberately not part of it, they change per experiment.
  - The pipeline gets one from the caller and applies it to `get_model()`.
  - The Hadamard weight is `layer_getter(model).weight`, so there is no accessor for it here.
"""

from typing import Protocol

import torch

from ..neuron_report import AttrFn
from ..presenters import ClusterPresenter
from ..sources import SampleSource


def reject_nnsight_wrapped(module: torch.nn.Module) -> None:
    if any(hasattr(submodule, "__nnsight_forward__") for submodule in module.modules()):
        raise RuntimeError(
            "the model is already wrapped in NNsight, which stores each forward at wrap time, so the "
            "AttnLRP patches would be silently skipped. Call patch_for_attn_lrp before wrapping."
        )


class ModelAdapter[SamplesT](Protocol):
    """`SamplesT` is what `make_source` consumes: a list of texts, a directory of DICOMs."""

    attr_fn: AttrFn
    """Explains one neuron at one position in pixel or embedding space, see `attribution`."""

    def get_model(self) -> torch.nn.Module:
        """The raw module the layer getters index into, for example `hf_model.model`.

        - The pipeline wraps it in NNsight for capture, after `patch_for_attn_lrp`.
        - Attribution uses it directly, which is why both see the same weights.
        """
        ...

    def patch_for_attn_lrp(self) -> None:
        """Call once, before the model is wrapped in NNsight.

        - NNsight stores each module's `forward` when it wraps the model, and calls that stored function afterwards.
        - The lxt patches replace `forward` on the class, so a model wrapped first never sees them.
        - Relevance is then silently computed with part of the AttnLRP rules missing.
          - On Gemma 3 the relevance norm was 30 times larger than with the rules applied.
        - `reject_nnsight_wrapped` raises if the order is wrong.
        - The patches are class-level, so they are process-wide.
        - Calling it twice is a no-op.
        """
        ...

    def make_source(self, samples: SamplesT) -> SampleSource: ...

    def make_presenter(self, source: SampleSource, clustered_label: str) -> ClusterPresenter: ...
