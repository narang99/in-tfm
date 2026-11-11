import torch
from jaxtyping import Bool, Float


def top_share_mask(
    values: Float[torch.Tensor, "*rows n"], share: float
) -> Bool[torch.Tensor, "*rows n"]:
    """Keeps, in each row, the largest positive terms making up `share` of the positive sum, and
    the same for the negative terms on their own."""
    return _top_share_of_positive(values, share) | _top_share_of_positive(
        -values, share
    )


def _top_share_of_positive(
    values: Float[torch.Tensor, "*rows n"], share: float
) -> Bool[torch.Tensor, "*rows n"]:
    """A term is kept while the terms larger than it still sum to less than the target.

    So the term that crosses the target is kept too, and a row with no positive terms keeps nothing.
    """
    positive = values.clamp(min=0)
    ordered, order = positive.sort(dim=-1, descending=True)
    sum_before = ordered.cumsum(dim=-1) - ordered
    target = share * positive.sum(dim=-1, keepdim=True)
    kept_in_order = (sum_before < target) & (ordered > 0)
    return torch.zeros_like(kept_in_order).scatter(-1, order, kept_in_order)
