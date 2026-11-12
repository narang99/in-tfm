"""A baseline keep: the same number of neurons per mask as the walk kept, chosen at random."""

from collections.abc import Iterator

import torch
from pydantic import BaseModel


def random_keep[KeepT: BaseModel](keep: KeepT, generator: torch.Generator) -> KeepT:
    """Every mask (one per kind per layer) keeps its count. Only its positions are shuffled."""
    shuffled = keep.model_copy(deep=True)
    shuffle_masks(shuffled, generator)
    return shuffled


def shuffle_masks(model: BaseModel, generator: torch.Generator) -> None:
    for name, mask in iter_masks(model):
        mask.copy_(shuffled_mask(mask, valid_slots(name, mask), generator))


def iter_masks(model: BaseModel) -> Iterator[tuple[str, torch.Tensor]]:
    """Every mask in a keep, by field name, through nested layers and per-layer lists."""
    for name, value in model:
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, BaseModel):
                yield from iter_masks(item)
            elif isinstance(item, torch.Tensor):
                yield name, item


def kept_shares(keep: BaseModel) -> dict[str, float]:
    """Kept over real neurons, per mask name, summed over layers."""
    kept: dict[str, int] = {}
    total: dict[str, int] = {}
    for name, mask in iter_masks(keep):
        valid = valid_slots(name, mask)
        kept[name] = kept.get(name, 0) + int(mask[valid].sum())
        total[name] = total.get(name, 0) + int(valid.sum())
    return {name: kept[name] / total[name] for name in kept}


def valid_slots(name: str, mask: torch.Tensor) -> torch.Tensor:
    """Attention weights on future tokens are always zero, so they are no real neuron to keep."""
    if name != "pattern":
        return torch.ones_like(mask)
    return torch.ones_like(mask).tril()


def shuffled_mask(
    mask: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    slots = valid.flatten().nonzero().flatten()
    n_kept = int(mask.flatten()[slots].sum())
    chosen = slots[torch.randperm(len(slots), generator=generator)[:n_kept]]
    shuffled = torch.zeros(mask.numel(), dtype=torch.bool)
    shuffled[chosen] = True
    return shuffled.view_as(mask)
