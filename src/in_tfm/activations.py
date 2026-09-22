"""Capturing per-token MLP activations via nnsight tracing.

`get_activations` streams samples from a `SampleSource` in batches and hooks one layer's
input/output through NNsight's `.trace()`, rather than running the whole model and reading
`outputs.hidden_states` - we need the *pre*-fc2 features (fc2's input) to build Hadamard
products downstream, and HF's stock output objects don't expose that.

Nothing here knows the modality: it asks the source for ids and batches, and returns tensors.
"""

import gc
from itertools import islice

import torch
from jaxtyping import Bool, Float
from nnsight.modeling.base import NNsight
from tqdm import tqdm

from .device import default_device, empty_cache
from .layers import LayerGetter
from .sources import SampleId, SampleSource


def clear_mem() -> None:
    """Drop Jupyter's cached Out/In history (if running under IPython) before freeing CUDA
    memory - `empty_cache()` alone can't reclaim tensors still referenced by cell
    outputs."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("reset", "out")
    except ImportError:
        pass
    empty_cache()
    gc.collect()


def get_activations(
    model: NNsight,
    source: SampleSource,
    layer_getter: LayerGetter,
    n_iter: int = 14,
    bs: int = 4,
    device: str | None = None,
) -> tuple[
    list[SampleId],
    Float[torch.Tensor, "batch seq hidden"],
    Float[torch.Tensor, "batch seq hidden"],
    Bool[torch.Tensor, "batch seq"],
]:
    """Inputs/outputs keep the (batch, seq_len, hidden) shape rather than flattening
    batch/seq_len together, so a later `argwhere` on the activations can be traced back to
    (batch_idx, token_idx) pairs.

    The validity mask rides along so hit selection can skip padding - see sources.ModelBatch.
    """
    device = device or default_device()
    it = iter(source.sample_ids())
    all_ids: list[SampleId] = []
    all_inputs: list[torch.Tensor] = []
    all_outputs: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    model = model.to(device)
    for _ in tqdm(range(n_iter)):
        ids = list(islice(it, bs))
        if not ids:
            break
        all_ids.extend(ids)
        batch = source.to_model_batch(ids).to(device)
        with model.trace(**batch.kwargs):
            lay = layer_getter(model)
            inputs = lay.input.save()
            outputs = lay.output.save()

        all_inputs.append(inputs.detach().cpu())
        all_outputs.append(outputs.detach().cpu())
        all_masks.append(batch.valid_mask.cpu())

        del inputs, outputs, batch
        empty_cache()
        gc.collect()

    return (
        all_ids,
        cat_padded(all_inputs),
        cat_padded(all_outputs),
        cat_padded(all_masks),
    )


def cat_padded(tensors: list[torch.Tensor]) -> torch.Tensor:
    # todo: this needs to be configurable in the function
    # a callable or something which right pads / left pads depending on the model
    """Concatenate along batch, right-padding the sequence axis to the widest batch.

    Each batch is tokenized independently and pads to its own longest sequence, so widths
    differ between batches even though they agree within one. The padding added here is marked
    invalid by the same mask (itself padded with False), so it never reaches a hit.
    """
    max_seq = max(t.shape[1] for t in tensors)
    padded = [
        torch.nn.functional.pad(t, (0,) * (2 * (t.ndim - 2)) + (0, max_seq - t.shape[1]))
        for t in tensors
    ]
    return torch.cat(padded, dim=0)
