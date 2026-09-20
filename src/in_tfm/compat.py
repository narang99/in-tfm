"""Process-wide monkeypatches needed for pinned third-party libraries to import or run at all -
not feature additions, just restoring things they expect from an older `transformers`.

Import this module (or anything that imports it) before
touching `lxt.efficient` or pruning attention heads.
"""

import torch
from transformers import pytorch_utils


def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    """Find the heads and flattened indices to keep, accounting for already-pruned heads.

    Restores a `transformers.pytorch_utils` function that newer `transformers` versions
    dropped. `lxt.efficient.models.bert` - imported eagerly by `lxt.efficient.core`, and so
    transitively by any `lxt.efficient` import at all - still calls it by attribute reference,
    so importing lxt breaks without this.
    """
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)
    heads = set(heads) - already_pruned_heads

    for head in heads:
        # Shift the head index left by however many smaller heads were already removed earlier.
        shifted_head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[shifted_head] = False

    index = torch.arange(n_heads * head_size)[mask.view(-1)].long()
    return heads, index


pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
