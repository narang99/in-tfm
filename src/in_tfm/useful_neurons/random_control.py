"""A baseline keep: the same number of neurons per mask as the walk kept, chosen at random."""

import torch
from pydantic import BaseModel


def random_keep[KeepT: BaseModel](keep: KeepT, generator: torch.Generator) -> KeepT:
    """Every mask (one per kind per layer) keeps its count. Only its positions are shuffled."""
    shuffled = keep.model_copy(deep=True)
    shuffle_masks(shuffled, generator)
    return shuffled


def shuffle_masks(model: BaseModel, generator: torch.Generator) -> None:
    for name, value in model:
        masks = value if isinstance(value, list) else [value]
        for mask in masks:
            if isinstance(mask, BaseModel):
                shuffle_masks(mask, generator)
            elif isinstance(mask, torch.Tensor):
                mask.copy_(shuffled_mask(mask, valid_slots(name, mask), generator))


def valid_slots(name: str, mask: torch.Tensor) -> torch.Tensor:
    """Attention weights on future tokens are always zero, so they are no real neuron to keep."""
    if name != "pattern":
        return torch.ones_like(mask)
    return torch.ones_like(mask).tril().expand_as(mask)


def shuffled_mask(
    mask: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    slots = valid.flatten().nonzero().flatten()
    n_kept = int(mask.flatten()[slots].sum())
    chosen = slots[torch.randperm(len(slots), generator=generator)[:n_kept]]
    shuffled = torch.zeros(mask.numel(), dtype=torch.bool)
    shuffled[chosen] = True
    return shuffled.view_as(mask)
