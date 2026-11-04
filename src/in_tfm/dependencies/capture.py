"""Every tap a dependency report reads, from one forward pass.

- Plain torch hooks, not NNsight, since the taps are read in whatever order the model runs them.
- Each tapped tensor goes through its layer's token layout, so all come back as `(batch, seq, hidden)`.
  - A conv's input tap is then its patch at each output cell, which is what its Hadamard rows need.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
from jaxtyping import Float
from pydantic import BaseModel

from ..sources import ModelBatch, SampleSource
from ..token_layout import TokenLayout
from .refs import Tap

LayoutFor = Callable[[torch.nn.Module], TokenLayout]
BatchCaptures = dict[Tap, Float[torch.Tensor, "batch seq hidden"]]


class CaptureSetup(BaseModel, arbitrary_types_allowed=True):
    model: torch.nn.Module
    source: SampleSource
    layout_for: LayoutFor
    batch_size: int
    device: str


def first_input(args: tuple) -> torch.Tensor:
    """Every tapped module takes the tensor it acts on as its first argument."""
    return args[0]


@contextmanager
def hooked(
    model: torch.nn.Module, taps: list[Tap]
) -> Generator[dict[Tap, torch.Tensor]]:
    saved: dict[Tap, torch.Tensor] = {}
    handles = []
    for tap in taps:
        module = model.get_submodule(tap.layer)
        if tap.port == "input":
            handles.append(
                module.register_forward_pre_hook(
                    lambda m, args, tap=tap: saved.__setitem__(tap, first_input(args))
                )
            )
        else:
            handles.append(
                module.register_forward_hook(
                    lambda m, args, out, tap=tap: saved.__setitem__(tap, out)
                )
            )
    try:
        yield saved
    finally:
        for handle in handles:
            handle.remove()


def in_layout(
    model: torch.nn.Module, tap: Tap, tensor: torch.Tensor, layout_for: LayoutFor
) -> Float[torch.Tensor, "batch seq hidden"]:
    layout = layout_for(model.get_submodule(tap.layer))
    return layout.inputs(tensor) if tap.port == "input" else layout.outputs(tensor)


def capture_taps(
    model: torch.nn.Module,
    batch: ModelBatch,
    taps: list[Tap],
    layout_for: LayoutFor,
    device: str,
) -> BatchCaptures:
    """Returned on the CPU, so a run holds at most one batch on the accelerator."""
    with torch.no_grad(), hooked(model, taps) as saved:
        model.to(device)(**batch.to(device).kwargs)
    return {
        tap: in_layout(model, tap, saved[tap].detach(), layout_for).cpu()
        for tap in taps
    }


def sample_captures(captures: BatchCaptures, row: int) -> dict[Tap, torch.Tensor]:
    return {tap: tensor[row] for tap, tensor in captures.items()}
